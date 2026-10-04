//! Runs the models on the CPU with candle: the subject model, a T5
//! encoder-decoder (CodeT5-small), and the type model, the same encoder
//! with a linear layer over its averaged states. Each is read from one GGUF
//! file that also holds its tokenizer.
//! The file keeps the matrices in 8-bit blocks (67 MB); they are expanded to
//! 32-bit floats when loaded (240 MB of memory), because candle multiplies
//! 8-bit blocks quickly only when built for AVX2, and even then the encoder
//! ran faster in floats.
//!
//! The network is written out here rather than taken from
//! candle-transformers, whose T5 differs from the one the model was trained
//! with (Hugging Face transformers): it scales the decoder's output by
//! √d_model instead of 1/√d_model, which changes the probabilities (the
//! confidence) though not the words, and its decoder uses the encoder's
//! (bidirectional) position buckets, which differ from the ninth token on.
//! Here the encoder's keys and values are also projected once per draft
//! instead of once per generated token.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::Linear;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use super::ModelDraft;

/// The encoder reads at most this many tokens, as in training.
const MAX_INPUT_TOKENS: usize = 512;
/// A subject is at most this many tokens, as in training.
const MAX_NEW_TOKENS: usize = 48;
/// Metadata key of the tokenizer (its tokenizer.json) in the GGUF file.
const TOKENIZER_KEY: &str = "tokenizer.huggingface.json";
/// Metadata key of the input layout the model was trained on; gca refuses
/// a model made for another layout.
pub const FORMAT_KEY: &str = "gca.draft.input_format";
pub const INPUT_FORMAT: u32 = 1;
/// The type model's input layout (input::type_format) and its types, in the
/// order of its outputs.
pub const TYPE_FORMAT_KEY: &str = "gca.type.input_format";
pub const TYPE_INPUT_FORMAT: u32 = 1;
const TYPES_KEY: &str = "gca.type.classes";
/// Settings a model file may carry for gca to use it with: the subject
/// model's lowest confidence for a draft to be offered, the type model's
/// weight against the built-in model.
const THRESHOLD_KEY: &str = "gca.draft.threshold";
const WEIGHT_KEY: &str = "gca.type.weight";

struct Config {
    d_model: usize,
    d_kv: usize,
    num_heads: usize,
    buckets: usize,
    max_distance: usize,
    eps: f64,
    decoder_start: u32,
    eos: u32,
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

struct EncoderLayer {
    norm: Tensor,
    attn: Attention,
    ff_norm: Tensor,
    wi: Linear,
    wo: Linear,
}

struct DecoderLayer {
    norm: Tensor,
    attn: Attention,
    cross_norm: Tensor,
    cross: Attention,
    ff_norm: Tensor,
    wi: Linear,
    wo: Linear,
}

/// A T5 encoder: its layers, its relative position biases (shared by the
/// layers) and its final norm.
struct Encoder {
    layers: Vec<EncoderLayer>,
    bias: Tensor,
    norm: Tensor,
}

/// The subject model and its tokenizer.
pub struct Drafter {
    cfg: Config,
    threshold: f32,
    shared: Tensor,
    lm_head: Linear,
    encoder: Encoder,
    decoder: Vec<DecoderLayer>,
    decoder_bias: Tensor,
    decoder_norm: Tensor,
    tokenizer: Tokenizer,
    device: Device,
}

struct Weights<'a> {
    content: &'a gguf_file::Content,
    file: std::fs::File,
    device: &'a Device,
}

impl Weights<'_> {
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.content
            .tensor(&mut self.file, name, self.device)
            .with_context(|| format!("the model has no usable {name}"))
    }

    fn float(&mut self, name: &str) -> Result<Tensor> {
        Ok(self.qtensor(name)?.dequantize(self.device)?)
    }

    fn matmul(&mut self, name: &str) -> Result<Linear> {
        Ok(Linear::new(self.float(name)?, None))
    }

    fn attention(&mut self, prefix: &str) -> Result<Attention> {
        Ok(Attention {
            q: self.matmul(&format!("{prefix}.q.weight"))?,
            k: self.matmul(&format!("{prefix}.k.weight"))?,
            v: self.matmul(&format!("{prefix}.v.weight"))?,
            o: self.matmul(&format!("{prefix}.o.weight"))?,
        })
    }

    fn encoder(&mut self, layers: usize) -> Result<Encoder> {
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("encoder.block.{i}.layer");
            blocks.push(EncoderLayer {
                norm: self.float(&format!("{p}.0.layer_norm.weight"))?,
                attn: self.attention(&format!("{p}.0.SelfAttention"))?,
                ff_norm: self.float(&format!("{p}.1.layer_norm.weight"))?,
                wi: self.matmul(&format!("{p}.1.DenseReluDense.wi.weight"))?,
                wo: self.matmul(&format!("{p}.1.DenseReluDense.wo.weight"))?,
            });
        }
        Ok(Encoder {
            layers: blocks,
            bias: self
                .float("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?,
            norm: self.float("encoder.final_layer_norm.weight")?,
        })
    }
}

/// Opens a model file and checks that it is of the kind wanted: its input
/// layout, under `format_key`, must be `format`.
fn open(
    path: &Path,
    format_key: &str,
    format: u32,
    kind: &str,
) -> Result<(gguf_file::Content, std::fs::File)> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("{} is not a model file gca can read: {e}", path.display()))?;
    if !content.metadata.contains_key(format_key) {
        bail!("{} is not a {kind} model file", path.display());
    }
    let found = meta_u32(&content, format_key)?;
    if found != format {
        bail!(
            "{} was made for another version of gca (input format {found}); \
             `gca model update` installs one for this gca",
            path.display()
        );
    }
    Ok((content, file))
}

fn read_config(content: &gguf_file::Content) -> Result<Config> {
    let u = |key: &str| meta_u32(content, &format!("t5.{key}")).map(|v| v as usize);
    Ok(Config {
        d_model: u("d_model")?,
        d_kv: u("d_kv")?,
        num_heads: u("num_heads")?,
        buckets: u("relative_attention_num_buckets")?,
        max_distance: u("relative_attention_max_distance")?,
        eps: content
            .metadata
            .get("t5.layer_norm_epsilon")
            .ok_or_else(|| anyhow!("the model file lacks t5.layer_norm_epsilon"))?
            .to_f32()
            .map_err(|e| anyhow!("t5.layer_norm_epsilon: {e}"))? as f64,
        decoder_start: u("decoder_start_token_id")? as u32,
        eos: u("eos_token_id")? as u32,
    })
}

/// The tokenizer stored in the model file, cutting inputs to what the
/// encoder reads.
fn read_tokenizer(content: &gguf_file::Content) -> Result<Tokenizer> {
    let json = content
        .metadata
        .get(TOKENIZER_KEY)
        .ok_or_else(|| anyhow!("the model file has no tokenizer"))?
        .to_string()
        .map_err(|e| anyhow!("{TOKENIZER_KEY}: {e}"))?;
    let mut tokenizer: Tokenizer = json
        .parse()
        .map_err(|e| anyhow!("the model's tokenizer is invalid: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_INPUT_TOKENS,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }))
        .map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

fn tokenize(tokenizer: &Tokenizer, input: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(input, true)
        .map_err(|e| anyhow!("could not tokenize: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

impl Encoder {
    /// The encoder's states, (len, d_model), for the embedded input.
    fn forward(&self, cfg: &Config, mut x: Tensor, device: &Device) -> Result<Tensor> {
        let len = x.dim(0)?;
        let buckets: Vec<u32> = (0..len)
            .flat_map(|i| (0..len).map(move |j| (i, j)))
            .map(|(i, j)| bucket(j as i64 - i as i64, true, cfg.buckets, cfg.max_distance))
            .collect();
        let bias = position_bias(&self.bias, buckets, len, len, cfg.num_heads, device)?;
        for l in &self.layers {
            let h = rms_norm(&x, &l.norm, cfg.eps)?;
            let q = heads(cfg, &l.attn.q.forward(&h)?)?;
            let k = heads(cfg, &l.attn.k.forward(&h)?)?;
            let v = heads(cfg, &l.attn.v.forward(&h)?)?;
            let a = attend(&q, &k, &v, Some(&bias))?;
            x = (x + l.attn.o.forward(&merge(&a)?)?)?;
            let h = rms_norm(&x, &l.ff_norm, cfg.eps)?;
            x = (x + l.wo.forward(&l.wi.forward(&h)?.relu()?)?)?;
        }
        rms_norm(&x, &self.norm, cfg.eps)
    }
}

/// An optional setting in the model file.
fn meta_f32(content: &gguf_file::Content, key: &str) -> Result<Option<f32>> {
    content
        .metadata
        .get(key)
        .map(|v| v.to_f32().map_err(|e| anyhow!("{key}: {e}")))
        .transpose()
}

fn meta_u32(content: &gguf_file::Content, key: &str) -> Result<u32> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| anyhow!("the model file lacks {key}"))?
        .to_u32()
        .map_err(|e| anyhow!("{key}: {e}"))
}

impl Drafter {
    /// Reads the model file written by [`convert`].
    pub fn load(path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let (content, file) = open(path, FORMAT_KEY, INPUT_FORMAT, "subject")?;
        let cfg = read_config(&content)?;
        let threshold = meta_f32(&content, THRESHOLD_KEY)?.unwrap_or(super::THRESHOLD);
        let u = |key: &str| meta_u32(&content, &format!("t5.{key}")).map(|v| v as usize);
        let (layers, decoder_layers) = (u("num_layers")?, u("num_decoder_layers")?);
        let tokenizer = read_tokenizer(&content)?;

        let mut w = Weights {
            content: &content,
            file,
            device: &device,
        };
        let shared = w.float("shared.weight")?;
        let lm_head = Linear::new(shared.clone(), None);
        let encoder = w.encoder(layers)?;
        let mut decoder = Vec::with_capacity(decoder_layers);
        for i in 0..decoder_layers {
            let p = format!("decoder.block.{i}.layer");
            decoder.push(DecoderLayer {
                norm: w.float(&format!("{p}.0.layer_norm.weight"))?,
                attn: w.attention(&format!("{p}.0.SelfAttention"))?,
                cross_norm: w.float(&format!("{p}.1.layer_norm.weight"))?,
                cross: w.attention(&format!("{p}.1.EncDecAttention"))?,
                ff_norm: w.float(&format!("{p}.2.layer_norm.weight"))?,
                wi: w.matmul(&format!("{p}.2.DenseReluDense.wi.weight"))?,
                wo: w.matmul(&format!("{p}.2.DenseReluDense.wo.weight"))?,
            });
        }
        Ok(Drafter {
            decoder_bias: w
                .float("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?,
            decoder_norm: w.float("decoder.final_layer_norm.weight")?,
            cfg,
            threshold,
            shared,
            lm_head,
            encoder,
            decoder,
            tokenizer,
            device,
        })
    }

    /// The input's token ids, cut to what the encoder reads.
    pub fn tokens(&self, input: &str) -> Result<Vec<u32>> {
        tokenize(&self.tokenizer, input)
    }

    /// The subject for a model input (see [`super::input::format`]), by
    /// greedy decoding, with the mean log-probability of its tokens and of
    /// the end token.
    pub fn draft(&self, input: &str) -> Result<ModelDraft> {
        Ok(self.generate(&self.tokens(input)?)?.draft)
    }

    /// Greedy decoding of the input's token ids.
    pub fn generate(&self, ids: &[u32]) -> Result<Generated> {
        let encoded = self.encode(ids)?;
        let mut cache = Cache {
            self_kv: vec![None; self.decoder.len()],
            cross_kv: self
                .decoder
                .iter()
                .map(|l| {
                    Ok((
                        self.heads(&l.cross.k.forward(&encoded)?)?,
                        self.heads(&l.cross.v.forward(&encoded)?)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let mut out = Vec::new();
        let mut token = self.cfg.decoder_start;
        let (mut log_prob, mut count) = (0.0f64, 0usize);
        for pos in 0..MAX_NEW_TOKENS {
            let logits: Vec<f32> = self.decode_step(token, pos, &mut cache)?.to_vec1()?;
            let (best, top) =
                logits
                    .iter()
                    .enumerate()
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc },
                    );
            let sum: f64 = logits.iter().map(|&v| ((v - top) as f64).exp()).sum();
            log_prob += -(sum.ln());
            count += 1;
            token = best as u32;
            if token == self.cfg.eos {
                break;
            }
            out.push(token);
        }
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|e| anyhow!("could not decode: {e}"))?;
        Ok(Generated {
            draft: ModelDraft {
                subject: super::input::py_strip(&text).to_string(),
                confidence: (log_prob / count.max(1) as f64) as f32,
                threshold: self.threshold,
                repeats: 0,
            },
            steps: count,
        })
    }

    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        heads(&self.cfg, x)
    }

    fn encode(&self, ids: &[u32]) -> Result<Tensor> {
        self.encoder
            .forward(&self.cfg, self.embed(ids)?, &self.device)
    }

    /// One decoder step for the token at `pos`: the logits of the next one.
    fn decode_step(&self, token: u32, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let mut x = self.embed(&[token])?;
        let buckets: Vec<u32> = (0..=pos)
            .map(|j| {
                bucket(
                    j as i64 - pos as i64,
                    false,
                    self.cfg.buckets,
                    self.cfg.max_distance,
                )
            })
            .collect();
        let bias = position_bias(
            &self.decoder_bias,
            buckets,
            1,
            pos + 1,
            self.cfg.num_heads,
            &self.device,
        )?;
        for (i, l) in self.decoder.iter().enumerate() {
            let h = rms_norm(&x, &l.norm, self.cfg.eps)?;
            let q = self.heads(&l.attn.q.forward(&h)?)?;
            let mut k = self.heads(&l.attn.k.forward(&h)?)?;
            let mut v = self.heads(&l.attn.v.forward(&h)?)?;
            if let Some((pk, pv)) = &cache.self_kv[i] {
                k = Tensor::cat(&[pk, &k], 1)?;
                v = Tensor::cat(&[pv, &v], 1)?;
            }
            let a = attend(&q, &k, &v, Some(&bias))?;
            cache.self_kv[i] = Some((k, v));
            x = (x + l.attn.o.forward(&merge(&a)?)?)?;
            let h = rms_norm(&x, &l.cross_norm, self.cfg.eps)?;
            let q = self.heads(&l.cross.q.forward(&h)?)?;
            let (ck, cv) = &cache.cross_kv[i];
            let a = attend(&q, ck, cv, None)?;
            x = (x + l.cross.o.forward(&merge(&a)?)?)?;
            let h = rms_norm(&x, &l.ff_norm, self.cfg.eps)?;
            x = (x + l.wo.forward(&l.wi.forward(&h)?.relu()?)?)?;
        }
        let x = rms_norm(&x, &self.decoder_norm, self.cfg.eps)?;
        // the output embedding is the input one: scale by 1/√d_model first
        let x = x.affine((self.cfg.d_model as f64).powf(-0.5), 0.0)?;
        Ok(self.lm_head.forward(&x)?.squeeze(0)?)
    }

    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        embed(&self.shared, ids, &self.device)
    }
}

/// The type model and its tokenizer: the encoder's states averaged over the
/// input's tokens, then a linear layer to the types.
pub struct Classifier {
    cfg: Config,
    weight: f64,
    shared: Tensor,
    encoder: Encoder,
    head: Linear,
    classes: Vec<String>,
    tokenizer: Tokenizer,
    device: Device,
}

impl Classifier {
    /// Reads the model file written by [`convert_classifier`].
    pub fn load(path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let (content, file) = open(path, TYPE_FORMAT_KEY, TYPE_INPUT_FORMAT, "type")?;
        let cfg = read_config(&content)?;
        let weight = meta_f32(&content, WEIGHT_KEY)?.map_or(super::TYPE_WEIGHT, f64::from);
        let layers = meta_u32(&content, "t5.num_layers")? as usize;
        let classes = content
            .metadata
            .get(TYPES_KEY)
            .ok_or_else(|| anyhow!("the model file lacks {TYPES_KEY}"))?
            .to_vec()
            .map_err(|e| anyhow!("{TYPES_KEY}: {e}"))?
            .iter()
            .map(|v| {
                v.to_string()
                    .cloned()
                    .map_err(|e| anyhow!("{TYPES_KEY}: {e}"))
            })
            .collect::<Result<Vec<String>>>()?;
        let tokenizer = read_tokenizer(&content)?;
        let mut w = Weights {
            content: &content,
            file,
            device: &device,
        };
        let shared = w.float("shared.weight")?;
        let encoder = w.encoder(layers)?;
        let head = Linear::new(w.float("head.weight")?, Some(w.float("head.bias")?));
        if head.weight().dim(0)? != classes.len() {
            bail!("the model file's types do not match its output layer");
        }
        Ok(Classifier {
            cfg,
            weight,
            shared,
            encoder,
            head,
            classes,
            tokenizer,
            device,
        })
    }

    /// The types, in the order of [`Classifier::probabilities`].
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// How much the model's probabilities count against the built-in
    /// model's, from 0 to 1.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// The input's token ids, cut to what the encoder reads.
    pub fn tokens(&self, input: &str) -> Result<Vec<u32>> {
        tokenize(&self.tokenizer, input)
    }

    /// The probability of each type for a model input (see
    /// [`super::input::type_format`]).
    pub fn probabilities(&self, input: &str) -> Result<Vec<f64>> {
        self.probabilities_of(&self.tokens(input)?)
    }

    /// The probability of each type for the input's token ids.
    pub fn probabilities_of(&self, ids: &[u32]) -> Result<Vec<f64>> {
        let states = self.encoder.forward(
            &self.cfg,
            embed(&self.shared, ids, &self.device)?,
            &self.device,
        )?;
        let pooled = states.mean_keepdim(0)?;
        let logits: Vec<f32> = self.head.forward(&pooled)?.squeeze(0)?.to_vec1()?;
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp: Vec<f64> = logits.iter().map(|&v| (v as f64 - top).exp()).collect();
        let sum: f64 = exp.iter().sum();
        Ok(exp.into_iter().map(|v| v / sum).collect())
    }
}

fn embed(shared: &Tensor, ids: &[u32], device: &Device) -> Result<Tensor> {
    let ids = Tensor::new(ids, device)?;
    Ok(shared.index_select(&ids, 0)?)
}

/// (len, d_model) -> (heads, len, d_kv)
fn heads(cfg: &Config, x: &Tensor) -> Result<Tensor> {
    let len = x.dim(0)?;
    Ok(x.reshape((len, cfg.num_heads, cfg.d_kv))?
        .transpose(0, 1)?
        .contiguous()?)
}

/// (heads, q_len, k_len) biases for the given buckets, in row order.
fn position_bias(
    table: &Tensor,
    buckets: Vec<u32>,
    q_len: usize,
    k_len: usize,
    num_heads: usize,
    device: &Device,
) -> Result<Tensor> {
    let idx = Tensor::from_vec(buckets, q_len * k_len, device)?;
    Ok(table
        .index_select(&idx, 0)?
        .reshape((q_len, k_len, num_heads))?
        .permute((2, 0, 1))?
        .contiguous()?)
}

/// What greedy decoding wrote.
pub struct Generated {
    pub draft: ModelDraft,
    /// Tokens generated, the end token included if it came.
    pub steps: usize,
}

/// The decoder's keys and values: its own, growing with each token, and the
/// encoder's, projected once.
struct Cache {
    self_kv: Vec<Option<(Tensor, Tensor)>>,
    cross_kv: Vec<(Tensor, Tensor)>,
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&(variance + eps)?.sqrt()?)?
        .broadcast_mul(weight)?)
}

/// T5 attention: no 1/√d scaling, an additive position bias.
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let mut scores = q.matmul(&k.t()?)?;
    if let Some(bias) = bias {
        scores = scores.broadcast_add(bias)?;
    }
    Ok(candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)?)
}

/// (heads, len, d_kv) -> (len, heads * d_kv)
fn merge(x: &Tensor) -> Result<Tensor> {
    let (heads, len, d) = x.dims3()?;
    Ok(x.transpose(0, 1)?.contiguous()?.reshape((len, heads * d))?)
}

/// T5's relative position bucket for key position minus query position, as
/// transformers computes it (in 32-bit floats).
fn bucket(relative: i64, bidirectional: bool, num_buckets: usize, max_distance: usize) -> u32 {
    let mut buckets = num_buckets as i64;
    let mut base = 0;
    let n = if bidirectional {
        buckets /= 2;
        if relative > 0 {
            base = buckets;
        }
        relative.abs()
    } else {
        -relative.min(0)
    };
    let max_exact = buckets / 2;
    let offset = if n < max_exact {
        n
    } else {
        let scale = ((max_distance as f64) / (max_exact as f64)).ln() as f32;
        let large = max_exact
            + ((n as f32 / max_exact as f32).ln() / scale * (buckets - max_exact) as f32) as i64;
        large.min(buckets - 1)
    };
    (base + offset) as u32
}

/// What [`convert`] wrote.
pub struct Converted {
    pub tensors: usize,
    pub quantized: usize,
    pub bytes: u64,
}

/// Writes a fine-tuned checkpoint (a transformers folder with config.json,
/// model.safetensors and tokenizer.json) as one GGUF file gca can load: the
/// matrices in 8-bit blocks (Q8_0) if `quantize`, the rest in 32-bit floats,
/// and the configuration and tokenizer as metadata.
pub fn convert(
    checkpoint: &Path,
    out: &Path,
    quantize: bool,
    threshold: Option<f32>,
) -> Result<Converted> {
    let (config, mut metadata) = checkpoint_metadata(checkpoint)?;
    metadata.insert(1, (FORMAT_KEY.into(), gguf_file::Value::U32(INPUT_FORMAT)));
    if let Some(t) = threshold {
        metadata.push((THRESHOLD_KEY.into(), gguf_file::Value::F32(t)));
    }
    if config["tie_word_embeddings"] == false {
        bail!("only T5 models with tied embeddings are supported");
    }
    let tensors: HashMap<String, Tensor> =
        candle_core::safetensors::load(checkpoint.join("model.safetensors"), &Device::Cpu)?;
    let tensors: Vec<(String, Tensor)> = tensors
        .into_iter()
        .filter(|(n, _)| n != "lm_head.weight")
        .collect();
    write_gguf(out, &metadata, tensors, quantize)
}

/// Writes a type model checkpoint (train_type.py's folder: the encoder as a
/// transformers T5EncoderModel, head.pt with the output layer and types.json)
/// as one GGUF file, as [`convert`] does; the output layer stays in 32-bit
/// floats.
pub fn convert_classifier(
    checkpoint: &Path,
    out: &Path,
    quantize: bool,
    weight: Option<f32>,
) -> Result<Converted> {
    let (_, mut metadata) = checkpoint_metadata(checkpoint)?;
    if let Some(w) = weight {
        if !(0.0..=1.0).contains(&w) {
            bail!("the weight must be from 0 to 1");
        }
        metadata.push((WEIGHT_KEY.into(), gguf_file::Value::F32(w)));
    }
    let types: Vec<String> = serde_json::from_str(
        &std::fs::read_to_string(checkpoint.join("types.json")).with_context(|| {
            format!("could not read {}", checkpoint.join("types.json").display())
        })?,
    )
    .context("types.json is not a list of types")?;
    metadata.insert(
        1,
        (
            TYPE_FORMAT_KEY.into(),
            gguf_file::Value::U32(TYPE_INPUT_FORMAT),
        ),
    );
    metadata.push((
        TYPES_KEY.into(),
        gguf_file::Value::Array(
            types
                .iter()
                .cloned()
                .map(gguf_file::Value::String)
                .collect(),
        ),
    ));
    let encoder: HashMap<String, Tensor> =
        candle_core::safetensors::load(checkpoint.join("model.safetensors"), &Device::Cpu)?;
    // the input embedding is `shared`; a separate copy of it may be saved too
    let mut tensors: Vec<(String, Tensor)> = encoder
        .into_iter()
        .filter(|(n, _)| n != "encoder.embed_tokens.weight")
        .collect();
    let head = candle_core::pickle::read_all(checkpoint.join("head.pt"))
        .with_context(|| format!("could not read {}", checkpoint.join("head.pt").display()))?;
    for (name, t) in head {
        tensors.push((format!("head.{name}"), t));
    }
    let names: Vec<&str> = tensors.iter().map(|(n, _)| n.as_str()).collect();
    for wanted in ["shared.weight", "head.weight", "head.bias"] {
        if !names.contains(&wanted) {
            bail!("the checkpoint lacks {wanted}");
        }
    }
    let rows = tensors
        .iter()
        .find(|(n, _)| n == "head.weight")
        .map(|(_, t)| t.dim(0))
        .transpose()?;
    if rows != Some(types.len()) {
        bail!("types.json does not match the output layer");
    }
    write_gguf(out, &metadata, tensors, quantize)
}

/// config.json and the metadata both kinds of model file carry: the
/// architecture, the tokenizer and the T5 configuration.
fn checkpoint_metadata(
    checkpoint: &Path,
) -> Result<(serde_json::Value, Vec<(String, gguf_file::Value)>)> {
    let read = |name: &str| {
        std::fs::read_to_string(checkpoint.join(name))
            .with_context(|| format!("could not read {}", checkpoint.join(name).display()))
    };
    let config: serde_json::Value = serde_json::from_str(&read("config.json")?)?;
    let tokenizer = read("tokenizer.json")?;
    tokenizer
        .parse::<Tokenizer>()
        .map_err(|e| anyhow!("tokenizer.json is invalid: {e}"))?;
    if config["model_type"] != "t5" || config["feed_forward_proj"] != "relu" {
        bail!("only T5 models with ReLU feed-forward layers are supported");
    }
    let num = |key: &str| -> Result<u32> {
        config[key]
            .as_u64()
            .map(|v| v as u32)
            .ok_or_else(|| anyhow!("config.json lacks {key}"))
    };
    let mut metadata: Vec<(String, gguf_file::Value)> = vec![
        (
            "general.architecture".into(),
            gguf_file::Value::String("t5".into()),
        ),
        (TOKENIZER_KEY.into(), gguf_file::Value::String(tokenizer)),
        (
            "t5.layer_norm_epsilon".into(),
            gguf_file::Value::F32(config["layer_norm_epsilon"].as_f64().unwrap_or(1e-6) as f32),
        ),
    ];
    for key in [
        "vocab_size",
        "d_model",
        "d_kv",
        "d_ff",
        "num_layers",
        "num_decoder_layers",
        "num_heads",
        "relative_attention_num_buckets",
        "relative_attention_max_distance",
        "decoder_start_token_id",
        "eos_token_id",
        "pad_token_id",
    ] {
        metadata.push((format!("t5.{key}"), gguf_file::Value::U32(num(key)?)));
    }
    Ok((config, metadata))
}

/// Writes the tensors, sorted by name, and the metadata as a GGUF file: the
/// matrices whose rows fill whole 8-bit blocks as Q8_0 if `quantize`
/// (except the type model's output layer), the rest as 32-bit floats.
fn write_gguf(
    out: &Path,
    metadata: &[(String, gguf_file::Value)],
    mut tensors: Vec<(String, Tensor)>,
    quantize: bool,
) -> Result<Converted> {
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut quantized = 0;
    let mut qtensors = Vec::with_capacity(tensors.len());
    for (name, t) in &tensors {
        let t = t.to_dtype(DType::F32)?;
        let q = if quantize
            && !name.starts_with("head.")
            && t.rank() == 2
            && t.dim(1)? % GgmlDType::Q8_0.block_size() == 0
        {
            quantized += 1;
            QTensor::quantize(&t, GgmlDType::Q8_0)?
        } else {
            QTensor::quantize(&t, GgmlDType::F32)?
        };
        qtensors.push(q);
    }
    let mut file = std::fs::File::create(out)
        .with_context(|| format!("could not create {}", out.display()))?;
    let meta: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let refs: Vec<(&str, &QTensor)> = tensors
        .iter()
        .map(|(n, _)| n.as_str())
        .zip(qtensors.iter())
        .collect();
    gguf_file::write(&mut file, &meta, &refs)?;
    Ok(Converted {
        tensors: tensors.len(),
        quantized,
        bytes: std::fs::metadata(out)?.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where transformers' buckets change (key minus query position, the
    /// bucket from there on), for 32 buckets and a distance of 128.
    const BIDIRECTIONAL: [(i64, u32); 30] = [
        (-90, 14),
        (-63, 13),
        (-45, 12),
        (-31, 11),
        (-22, 10),
        (-15, 9),
        (-11, 8),
        (-7, 7),
        (-6, 6),
        (-5, 5),
        (-4, 4),
        (-3, 3),
        (-2, 2),
        (-1, 1),
        (0, 0),
        (1, 17),
        (2, 18),
        (3, 19),
        (4, 20),
        (5, 21),
        (6, 22),
        (7, 23),
        (8, 24),
        (12, 25),
        (16, 26),
        (23, 27),
        (32, 28),
        (46, 29),
        (64, 30),
        (91, 31),
    ];
    const UNIDIRECTIONAL: [(i64, u32); 31] = [
        (-112, 30),
        (-98, 29),
        (-86, 28),
        (-76, 27),
        (-66, 26),
        (-58, 25),
        (-51, 24),
        (-45, 23),
        (-39, 22),
        (-34, 21),
        (-30, 20),
        (-26, 19),
        (-23, 18),
        (-20, 17),
        (-18, 16),
        (-15, 15),
        (-14, 14),
        (-13, 13),
        (-12, 12),
        (-11, 11),
        (-10, 10),
        (-9, 9),
        (-8, 8),
        (-7, 7),
        (-6, 6),
        (-5, 5),
        (-4, 4),
        (-3, 3),
        (-2, 2),
        (-1, 1),
        (0, 0),
    ];

    fn check(changes: &[(i64, u32)], first: u32, bidirectional: bool) {
        let mut expected = first;
        let mut next = changes.iter().peekable();
        for rel in -2048..=2048 {
            if let Some(&&(at, b)) = next.peek() {
                if rel == at {
                    expected = b;
                    next.next();
                }
            }
            assert_eq!(
                bucket(rel, bidirectional, 32, 128),
                expected,
                "relative position {rel}"
            );
        }
    }

    #[test]
    fn position_buckets_match_transformers() {
        check(&BIDIRECTIONAL, 15, true);
        check(&UNIDIRECTIONAL, 31, false);
    }
}
