https://github.com/nicksenger/glowstick

Repository: nicksenger/glowstick
Files analyzed: 86

Estimated tokens: 109.2k

Directory structure:
└── nicksenger-glowstick/
    ├── README.md
    ├── Cargo.toml
    ├── LICENSE
    ├── examples/
    │   ├── burn-llama/
    │   │   ├── README.md
    │   │   ├── Cargo.toml
    │   │   └── src/
    │   │       ├── cache.rs
    │   │       ├── lib.rs
    │   │       ├── llama.rs
    │   │       ├── pretrained.rs
    │   │       ├── sampling.rs
    │   │       ├── shape.rs
    │   │       ├── transformer.rs
    │   │       ├── bin/
    │   │       │   └── chat.rs
    │   │       └── tokenizer/
    │   │           ├── base.rs
    │   │           ├── mod.rs
    │   │           └── tiktoken.rs
    │   └── candle-llama/
    │       ├── README.md
    │       ├── Cargo.toml
    │       └── src/
    │           ├── llama.rs
    │           ├── main.rs
    │           └── shape.rs
    ├── glowstick-burn/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       ├── tensor.rs
    │       └── op/
    │           ├── argmax.rs
    │           ├── cat.rs
    │           ├── expand.rs
    │           ├── flatten.rs
    │           ├── gather.rs
    │           ├── log_softmax.rs
    │           ├── matmul.rs
    │           ├── mean_dim.rs
    │           ├── mod.rs
    │           ├── narrow.rs
    │           ├── reshape.rs
    │           ├── softmax.rs
    │           ├── sort_descending_with_indices.rs
    │           ├── squeeze.rs
    │           ├── transpose.rs
    │           ├── tril_mask.rs
    │           ├── unsqueeze.rs
    │           └── var_mean.rs
    ├── glowstick-candle/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lib.rs
    │       ├── tensor.rs
    │       └── op/
    │           ├── broadcast_add.rs
    │           ├── cat.rs
    │           ├── conv.rs
    │           ├── expand.rs
    │           ├── flatten.rs
    │           ├── gather.rs
    │           ├── log_softmax.rs
    │           ├── matmul.rs
    │           ├── mod.rs
    │           ├── narrow.rs
    │           ├── narrow_dyn.rs
    │           ├── narrow_dyn_start.rs
    │           ├── reshape.rs
    │           ├── softmax.rs
    │           ├── squeeze.rs
    │           ├── transpose.rs
    │           └── unsqueeze.rs
    ├── src/
    │   ├── cmp.rs
    │   ├── diagnostic.rs
    │   ├── dynamic.rs
    │   ├── lib.rs
    │   ├── num.rs
    │   └── op/
    │       ├── broadcast.rs
    │       ├── cat.rs
    │       ├── cat_dyn.rs
    │       ├── convolution.rs
    │       ├── flatten.rs
    │       ├── gather.rs
    │       ├── matmul.rs
    │       ├── mod.rs
    │       ├── narrow.rs
    │       ├── narrow_dyn.rs
    │       ├── narrow_dyn_start.rs
    │       ├── pad.rs
    │       ├── permute.rs
    │       ├── reshape.rs
    │       ├── squeeze.rs
    │       ├── stack.rs
    │       ├── transpose.rs
    │       └── unsqueeze.rs
    └── .github/
        └── workflows/
            └── ci.yml


================================================
FILE: README.md
================================================
# glowstick

This crate makes working with tensors in Rust safe, **easy**, and _fun_ by tracking their shapes in the type system!

Example usage with candle:

```rust
use candle::{DType, Device};  
use glowstick::{Shape2, num::{U1, U2}, debug_tensor};
use glowstick_candle::{Tensor, matmul};

let a: Tensor<Shape2<U2, U1>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("tensor A");
let b: Tensor<Shape2<U1, U2>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("tensor B");

let c = matmul!(a, b).expect("matmul");
//debug_tensor!(c); // Compile error: [glowstick shape]: (RANK<_2>, (DIM<_2>, DIM<_2>))
```

Several operations are available:

```rust
use candle::{DType, Device};  
use glowstick::{num::{U0, U1, U2, U4, U3, U64, U5, U8}, Shape2, Shape4};
use glowstick_candle::{Tensor, conv2d, squeeze, unsqueeze, narrow, reshape, transpose, flatten, broadcast_add};

#[allow(unused)]
use glowstick::debug_tensor;

let my_tensor: Tensor<Shape2<U8, U8>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("tensor");
//debug_tensor!(my_tensor); // Compile error: [glowstick shape]: (RANK<_2>, (DIM<_8>, DIM<_8>))

let reshaped = reshape!(my_tensor, [U64]).expect("reshape"); 
//debug_tensor!(reshaped); // [glowstick shape]: (RANK<_1>, (DIM<_64>))

let unsqueezed = unsqueeze!(reshaped, U0, U2).expect("unsqueeze");
//debug_tensor!(unsqueezed); // [glowstick shape]: (RANK<_3>, (DIM<_1>, DIM<_64>, DIM<_1>))

let squeezed = squeeze!(unsqueezed, U0, U2).expect("squeeze");
//debug_tensor!(squeezed); // [glowstick shape]: (RANK<_1>, (DIM<_64>))

let narrowed = narrow!(squeezed, U0: [U8, U5]).expect("narrow");
//debug_tensor!(narrowed); // [glowstick shape]: (RANK<_1>, (DIM<_5>))

let expanded = broadcast_add!(Tensor::<Shape4<U2, U5, U2, U1>>::zeros(DType::F32, &Device::Cpu).unwrap(), narrowed).expect("add");
//debug_tensor!(expanded); // [glowstick shape]: (RANK<_4>, (DIM<_2>, DIM<_5>, DIM<_2>, DIM<_5>))

let swapped = transpose!(expanded, U1: U2).expect("swap");
//debug_tensor!(swapped); // [glowstick shape]: (RANK<_2>, (DIM<_2>, DIM<_5>, DIM<_5>))

let kernel: Tensor<Shape4<U4, U2, U3, U3>> = Tensor::zeros(DType::F32, &Device::Cpu).expect("kernel");
let conv = conv2d!(swapped, kernel, U0, U1, U1, 1).expect("conv2d");
//debug_tensor!(conv); // [glowstick shape]: (RANK<_4>, (DIM<_2>, DIM<_4>, DIM<_3>, DIM<_3>))

let flattened = flatten!(conv, [U1, U2]).expect("flatten");
//debug_tensor!(swapped); // [glowstick shape]: (RANK<_3>, (DIM<_2>, DIM<_12>, DIM<_3>))

assert_eq!(flattened.inner().dims(), [2, 12, 3]);
```

For examples of more extensive usage and integration with popular Rust ML frameworks like [candle](https://github.com/huggingface/candle) and [Burn](https://github.com/tracel-ai/burn), check out the examples directory.

The project is currently pre-1.0: breaking changes will be made!

## Features

- [x] Express tensor shapes as types
- [x] Support for dynamic dimensions (gradual typing)
- [x] Human-readable error messages (sort of)
- [x] Manually check type-level shapes (`debug_tensor!(_)`)
- [ ] Support for all ONNX operations




================================================
FILE: Cargo.toml
================================================
[package]
name = "glowstick"
description = "Gradual typing for tensor shapes"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
categories.workspace = true
keywords.workspace = true
rust-version.workspace = true

[dependencies]
typosaurus = { version = "0.2.0" }

[workspace]
members = [
  "glowstick-burn",
  "glowstick-candle",
  "examples/*",
]

[workspace.package]
version = "0.2.0"
authors = ["Nick Senger <dev@nsenger.com>"]
edition = "2024"
license = "MIT"
repository = "https://github.com/nicksenger/glowstick"
categories = ["science"]
keywords = ["science", "math", "machine-learning", "metaprogramming", "types"]
rust-version = "1.85"

[workspace.dependencies]
glowstick = { path = ".", version = "0.2.0" }
glowstick-burn = { path = "./glowstick-burn", version = "0.2.0" }
glowstick-candle = { path = "./glowstick-candle", version = "0.2.0" }
thiserror = "2"




================================================
FILE: LICENSE
================================================
Copyright 2025 Nick Senger

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



================================================
FILE: examples/burn-llama/README.md
================================================
## burn-llama

This example implements Meta's Llama 3.2 architecture using the [burn](https://github.com/tracel-ai/burn) framework, leveraging glowstick where possible for compile-time tensor shapes. It was largely copied from the corresponding [burn llama example](https://github.com/tracel-ai/models/tree/main/llama-burn).

Use the following command to test using Llama 3.2 1B:

`cargo run --release`

Note that most of the typed shape usage can be found in the model implementation (`src/transformer.rs`).




================================================
FILE: examples/burn-llama/Cargo.toml
================================================
[package]
name = "burn-llama"
edition = "2021"
default-run = "chat"

[features]
default = ["wgpu"]
3b = [] # loads llama3.2 3b instead of 1b
cuda = ["burn/cuda"]
wgpu = ["burn/wgpu"]

[dependencies]
base64 = { version = "0.22" }
burn = { version = "0.17", default-features = false, features = ["network", "std"] }
clap = { version = "4.5", features = ["derive"] }
dirs = { version = "5.0" }
glowstick = { path = "../.." }
glowstick-burn = { path = "../../glowstick-burn" }
rand = { version = "0.9" }
rustc-hash = { version = "1.1" }
thiserror = "2.0"
tiktoken-rs = { version = "0.5" }

[[bin]]
name = "chat"



================================================
FILE: examples/burn-llama/src/cache.rs
================================================
use burn::tensor::{backend::Backend as BurnBackend, Device, Tensor as BurnTensor};

use crate::shape::*;

pub(crate) struct AutoregressiveCache<Backend: BurnBackend> {
    /// Tensor cache with shape `[batch_size, num_heads, seq_len, d_model]`
    cache: BurnTensor<Backend, 4>,
    pub(crate) max_seq_len: usize,
    cur_seq_len: usize,
}

impl<Backend: BurnBackend> AutoregressiveCache<Backend> {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<Backend>,
    ) -> Self {
        Self {
            cache: BurnTensor::empty([max_batch_size, num_heads, max_seq_len, d_model], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = BurnTensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    pub fn forward(
        &mut self,
        tensor: Rank4Tensor<B, K, N, H, Backend>,
    ) -> Rank4Tensor<B, K, N, H, Backend> {
        let [batch_size, num_heads, seq_len, d_model] = tensor.dims();
        let mut new_seq_len = self.cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..d_model,
            ]);
            self.cache = self.cache.clone().slice_assign(
                [0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        self.cache = self.cache.clone().slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..d_model,
            ],
            tensor.into_inner(),
        );

        self.cur_seq_len += seq_len;

        self.cache
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model])
            .try_into()
            .unwrap()
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}



================================================
FILE: examples/burn-llama/src/lib.rs
================================================
pub(crate) mod cache;
pub mod llama;
pub mod pretrained;
pub mod sampling;
pub mod shape;
pub mod tokenizer;
mod transformer;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Glowstick error: {0}")]
    Glowstick(#[from] glowstick_burn::Error),
}



================================================
FILE: examples/burn-llama/src/llama.rs
================================================
use std::time::Instant;

use burn::{
    config::Config,
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    record::{FileRecorder, HalfPrecisionSettings, RecorderError},
    tensor::{
        backend::Backend as BurnBackend, cast::ToElement, Device, ElementConversion, Shape,
        Tensor as BurnTensor, TensorData,
    },
};
use glowstick::num::{Unsigned, U0, U1};
use glowstick_burn::{cat, narrow, reshape, softmax, squeeze};

use crate::{
    sampling::Sampler,
    shape::*,
    tokenizer::{Tiktoken, Tokenizer},
    transformer::{KeyValueCache, Transformer, TransformerConfig},
    Error,
};

use crate::pretrained::{self, ModelMeta};

#[derive(Config, Debug)]
pub struct LlamaConfig {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "32")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Rotary positional encoding (RoPE).
    #[config(default = "RopeConfig::new(10000.0)")]
    pub rope: RopeConfig,
    /// Maximum sequence length for input text.
    #[config(default = "128")]
    pub max_seq_len: usize,
    /// Maximum batch size (used for key-value cache).
    #[config(default = "1")]
    pub max_batch_size: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

/// Rotary positional encoding (RoPE)
#[derive(Config, Debug)]
pub struct RopeConfig {
    pub theta: f32,
    #[config(default = "None")]
    pub scaled: Option<RopeFrequencyScaling>,
}

/// RoPE frequency scaling.
#[derive(Config, Debug)]
pub struct RopeFrequencyScaling {
    #[config(default = "8.")]
    pub scale_factor: f32,
    #[config(default = "1.")]
    pub low_freq_factor: f32,
    #[config(default = "4.")]
    pub high_freq_factor: f32,
    #[config(default = "8192.")]
    pub old_context_len: f32,
}

impl LlamaConfig {
    pub fn with_tokenizer(tokenizer_path: &str) -> Self {
        Self::new(
            <F as Unsigned>::USIZE,
            <C as Unsigned>::USIZE,
            tokenizer_path.to_string(),
        )
        .with_d_model(<S as Unsigned>::USIZE)
        .with_num_attention_heads(<A as Unsigned>::USIZE)
        .with_num_hidden_layers(NUM_HIDDEN_LAYERS)
        .with_num_key_value_heads(Some(<K as Unsigned>::USIZE))
        .with_rope(
            RopeConfig::new(500000.0)
                .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
        )
    }

    /// Initialize a new [Llama](Llama) module.
    pub fn init<Backend: BurnBackend, T: Tokenizer>(
        &self,
        device: &Device<Backend>,
    ) -> Result<Llama<Backend, T>, String> {
        let tokenizer = T::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps)
        .init(device);

        let cache = (0..self.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    self.max_batch_size,
                    num_key_value_heads,
                    self.max_seq_len,
                    self.d_model / self.num_attention_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(self.rope.theta);

        let rope = if let Some(scaling) = &self.rope.scaled {
            let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);
            rope.init_with_frequency_scaling(freq_scaling_fn, device)
        } else {
            rope.init(device)
        };

        Ok(Llama {
            tokenizer,
            model,
            cache,
            rope,
            device: device.clone(),
        })
    }
    pub fn load_llama<Backend: BurnBackend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<Backend>,
    ) -> Result<Llama<Backend, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::with_tokenizer(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<Backend, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.2 model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    pub fn llama3_2_pretrained<Backend: BurnBackend>(
        max_seq_len: usize,
        device: &Device<Backend>,
    ) -> Result<Llama<Backend, Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        #[cfg(not(feature = "3b"))]
        let model = pretrained::Llama::Llama321bInstruct.pretrained();
        #[cfg(feature = "3b")]
        let model = pretrained::Llama::Llama323bInstruct.pretrained();

        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }
}

fn check_context_length(max_seq_len: usize, max_context_len: usize) {
    assert!(
        max_seq_len <= max_context_len,
        "Maximum sequence length must not exceed {max_context_len}"
    );
}

/// Generated text sample output.
pub struct GenerationOutput {
    /// The generated text.
    pub text: String,
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: f64,
}

/// Meta Llama large language model and tokenizer.
pub struct Llama<Backend: BurnBackend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Llama decoder-only transformer.
    pub model: Transformer<Backend>,
    /// Key-value cache for each transformer block.
    pub cache: Vec<KeyValueCache<Backend>>,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<Backend>,
    pub device: Device<Backend>,
}

impl<Backend: BurnBackend, T: Tokenizer> Llama<Backend, T> {
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
    ) -> Result<GenerationOutput, Error> {
        let mut tokens = self.tokenize(prompt);
        let prompt_len = tokens.dims()[0];
        let stop_tokens = BurnTensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);

        let mut num_tokens: usize = 0;
        let mut input_pos =
            Rank1IntTensor::<N, Backend>::arange(0..prompt_len as i64, &self.device);
        let now = Instant::now();
        for i in 0..sample_len {
            let ctx = if i == 0 { prompt_len } else { 1 };
            let x = narrow!(tokens.clone(), U0: [if i == 0 { 0 } else { prompt_len + num_tokens - 1 }, { ctx }] => N);
            let x = reshape!(x, [B, { ctx as i32 } => N]);
            let logits: Rank3Tensor<B, N, C, Backend> =
                self.model.forward(x, &mut self.cache, &self.rope)?;

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let next_token_logits = narrow!(
                logits,
                U0: [{ 0 }, { batch_size }] => B,
                U1: [{ seq_len - 1 }, U1]
            );
            let mut next_token_logits = squeeze!(next_token_logits, U1);

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let sampled = sampler.sample(next_token_logits);
            let next_token = squeeze!(sampled, U0);

            // Stop when any of the valid stop tokens is encountered
            if stop_tokens
                .clone()
                .equal(next_token.clone().into_inner())
                .any()
                .into_scalar()
                .to_bool()
            {
                break;
            }

            // Update with the new generated token
            tokens = cat!(vec![tokens, next_token.into_inner().try_into()?], U0 => N);
            num_tokens += 1;

            // Advance
            let t = input_pos.dims()[0];
            input_pos = narrow!(input_pos, U0: [{ t - 1 }, { 1 }] => N);
        }

        let tokens = tokens.into_data().as_slice::<Backend::IntElem>().unwrap()
            [prompt_len..prompt_len + num_tokens]
            .iter()
            .map(|t| t.elem::<u32>())
            .collect::<Vec<_>>();

        let generated = self.tokenizer.decode(tokens);
        let elapsed = now.elapsed().as_secs_f64();

        Ok(GenerationOutput {
            text: generated,
            tokens: num_tokens,
            time: elapsed,
        })
    }

    /// Encode a string into a tensor of tokens.
    fn tokenize(&self, text: &str) -> Rank1IntTensor<N, Backend> {
        let tokens = self.tokenizer.encode(text, true, false);

        let shape = Shape::new([tokens.len()]);
        Rank1IntTensor::<N, Backend>::from_ints(TensorData::new(tokens, shape), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder<Backend>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {}s", elapsed);

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder<Backend>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        println!("Loading record...");
        let now = Instant::now();
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        Ok(self)
    }

    /// Reset the model state (used between generations)
    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|cache| cache.reset());
    }
}

impl RopeFrequencyScaling {
    /// Applies frequency scaling by parts following Llama 3.1's scheme.
    ///
    /// Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45
    pub fn freq_scaling_by_parts<Backend: BurnBackend>(
        &self,
        freqs: BurnTensor<Backend, 1>,
    ) -> BurnTensor<Backend, 1> {
        let low_freq_wavelen = self.old_context_len / self.low_freq_factor;
        let high_freq_wavelen = self.old_context_len / self.high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(self.old_context_len)
            .sub_scalar(self.low_freq_factor)
            .div_scalar(self.high_freq_factor - self.low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(self.scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(self.scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);
        new_freqs.mask_where(cond, freqs)
    }
}

pub(crate) fn temperature_scaled_softmax<Backend: BurnBackend>(
    logits: Rank2Tensor<B, C, Backend>,
    temperature: f64,
) -> Rank2Tensor<B, C, Backend> {
    softmax!(logits / temperature, U1)
}



================================================
FILE: examples/burn-llama/src/pretrained.rs
================================================
/// Pre-trained model metadata.
pub struct Pretrained {
    pub(super) name: &'static str,
    pub(super) model: &'static str,
    pub(super) tokenizer: &'static str,
}

mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::PathBuf;

    impl Pretrained {
        /// Download the file to the local cache directory.
        fn download(&self, url: &str) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("llama-burn")
                .join(self.name);

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = url
                .rsplit_once('/')
                .unwrap()
                .1
                .replace("?download=true", "");
            let file_name = model_dir.join(&file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(url, &file_base_name);

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                output_file.write_all(&bytes)?; // write_all is not OS limited (files over 2GB)
            }

            Ok(file_name)
        }

        /// Download the pre-trained model weights to the local cache directory.
        pub fn download_weights(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.model)
        }

        /// Download the tokenizer to the local cache directory.
        pub fn download_tokenizer(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.tokenizer)
        }
    }
}

pub trait ModelMeta {
    fn pretrained(&self) -> Pretrained;
}

/// Llama pre-trained weights.
pub enum Llama {
    Llama321bInstruct,
    Llama323bInstruct,
}

impl ModelMeta for Llama {
    fn pretrained(&self) -> Pretrained {
        match self {
            Self::Llama321bInstruct => Pretrained {
                name: "Llama-3.2-1B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama323bInstruct => Pretrained {
                name: "Llama-3.2-3B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
        }
    }
}



================================================
FILE: examples/burn-llama/src/sampling.rs
================================================
use burn::tensor::backend::Backend as BurnBackend;
use glowstick::num::{U0, U1};
use glowstick_burn::{argmax, narrow, sort_descending_with_indices};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rngs::StdRng,
    SeedableRng,
};

use crate::shape::*;

pub enum Sampler {
    TopP(TopP),
    Argmax,
}

impl Sampler {
    pub fn sample<Backend: BurnBackend>(
        &mut self,
        logits: Rank2Tensor<B, C, Backend>,
    ) -> Rank2IntTensor<B, U1, Backend> {
        match self {
            Self::TopP(s) => s.sample(logits),
            Self::Argmax => {
                argmax!(logits, U1)
            }
        }
    }
}

pub trait Sampling {
    fn sample<Backend: BurnBackend>(
        &mut self,
        logits: Rank2Tensor<B, C, Backend>,
    ) -> Rank2IntTensor<B, U1, Backend>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { p, rng }
    }
}

impl Sampling for TopP {
    fn sample<Backend: BurnBackend>(
        &mut self,
        probs: Rank2Tensor<B, C, Backend>,
    ) -> Rank2IntTensor<B, U1, Backend> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );
        let (probs_sort, probs_idx) = sort_descending_with_indices!(probs, U1);

        // TODO: cumsum + Distribution::Multinomial support

        let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();

        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });

        let next_token_idx = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng);

        narrow!(probs_idx, U0: [{ 0 }, { 1 }] => B, U1: [{ next_token_idx }, { 1 }] => U1)
    }
}



================================================
FILE: examples/burn-llama/src/shape.rs
================================================
use std::ops::{Add, Div, Mul};

use burn::tensor::{Int, Tensor as BurnTensor};
use glowstick::num::{U1, U1000, U128, U296};
use glowstick::{dyndims, Shape1, Shape2, Shape3, Shape4};
use glowstick_burn::Tensor;

dyndims! {
    N: SequenceLength
}

pub type Rank1Tensor<D1, B> = Tensor<BurnTensor<B, 1>, Shape1<D1>>;
pub type Rank2Tensor<D1, D2, B> = Tensor<BurnTensor<B, 2>, Shape2<D1, D2>>;
pub type Rank3Tensor<D1, D2, D3, B> = Tensor<BurnTensor<B, 3>, Shape3<D1, D2, D3>>;
pub type Rank4Tensor<D1, D2, D3, D4, B> = Tensor<BurnTensor<B, 4>, Shape4<D1, D2, D3, D4>>;
pub type Rank1IntTensor<D1, B> = Tensor<BurnTensor<B, 1, Int>, Shape1<D1>>;
pub type Rank2IntTensor<D1, D2, B> = Tensor<BurnTensor<B, 2, Int>, Shape2<D1, D2>>;

// TODO: support batched inference here like in the candle example
pub type B = U1;

pub type C = <<U128 as Mul<U1000>>::Output as Add<U296>>::Output;
pub type H = <S as Div<A>>::Output; // Head-dim
pub type Q = S;
pub type KV = <<S as Div<A>>::Output as Mul<K>>::Output;
pub type R = <A as Div<K>>::Output;

#[cfg(not(feature = "3b"))]
mod config_dims {
    use glowstick::num::{U2048, U32, U8, U8192};

    pub type A = U32; // Attention Heads
    pub type K = U8; // Key-Value Heads
    pub type S = U2048; // Hidden Size
    pub type F = U8192; // Feed-Forward Length
    pub const NUM_HIDDEN_LAYERS: usize = 16;
    pub const ROPE_THETA: f32 = 500000.;
}

#[cfg(feature = "3b")]
mod config_dims {
    use glowstick::num::{U10, U24, U300, U72, U8, U8192};

    type U3072 = <<U300 as std::ops::Mul<U10>>::Output as std::ops::Add<U72>>::Output;
    pub type A = U24; // Attention Heads
    pub type K = U8; // Key-Value Heads
    pub type S = U3072; // Hidden Size
    pub type F = U8192; // Feed-Forward Length
    pub const NUM_HIDDEN_LAYERS: usize = 28;
    pub const ROPE_THETA: f32 = 500000.;
}

pub use config_dims::*;



================================================
FILE: examples/burn-llama/src/transformer.rs
================================================
use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
        SwiGlu, SwiGluConfig,
    },
    tensor::{backend::Backend, Device},
};
use glowstick::num::{U1, U2, U3};
use glowstick_burn::{expand, matmul, reshape, softmax, transpose, tril_mask, unsqueeze};

// Using BS for Batch-Size here, as Burn's `Module` proc-macro expects B for the backend
use crate::cache::AutoregressiveCache;
use crate::shape::{Rank2IntTensor, Rank3Tensor, Rank4Tensor, A, B as BS, C, H, K, KV, N, Q, R, S};

/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
}

impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Transformer<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}

/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
    output: Linear<B>,
}

impl<Backend: burn::tensor::backend::Backend> Transformer<Backend> {
    pub fn forward(
        &self,
        input: Rank2IntTensor<BS, N, Backend>,
        cache: &mut Vec<KeyValueCache<Backend>>,
        rope: &RotaryEncoding<Backend>,
    ) -> Result<Rank3Tensor<BS, N, C, Backend>, crate::Error> {
        let mut h: Rank3Tensor<BS, N, S, Backend> =
            self.tok_embeddings.forward(input.into_inner()).try_into()?;

        for (layer, cache) in self.layers.iter().zip(cache.into_iter()) {
            h = layer.forward(h, cache, rope).unwrap();
        }

        let h = self.norm.forward(h.into_inner());
        Ok(self.output.forward(h).try_into()?)
    }
}

/// Configuration to create a [decoder-only transformer block](TransformerBlock).
#[derive(Config)]
pub struct TransformerBlockConfig {
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Initialize a new [decoder-only transformer block](TransformerBlock).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerBlock<B> {
        let attention =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_kv_heads).init(device);
        let feed_forward = FeedForwardConfig::new(self.d_model, self.hidden_size).init(device);
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }
}

/// Decoder-only transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention.
    attention: MultiHeadAttention<B>,
    /// Feed-forward transformation.
    feed_forward: FeedForward<B>,
    /// Attention pre-normalization.
    attention_norm: RmsNorm<B>,
    /// Feed-forward pre-normalization.
    ffn_norm: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        input: Rank3Tensor<BS, N, S, B>,
        cache: &mut KeyValueCache<B>,
        rope: &RotaryEncoding<B>,
    ) -> Result<Rank3Tensor<BS, N, S, B>, crate::Error> {
        let h: Rank3Tensor<BS, N, S, B> = input.clone()
            + self.attention.forward(
                self.attention_norm.forward(input.into_inner()).try_into()?,
                cache,
                rope,
            )?;
        let y: Rank3Tensor<BS, N, S, B> = self
            .feed_forward
            .forward(self.ffn_norm.forward(h.clone().into_inner()).try_into()?)?;
        Ok(h + y)
    }
}

/// Configuration to create a [feed-forward transformation network](FeedForward).
#[derive(Config)]
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let swiglu = SwiGluConfig::new(self.d_model, self.hidden_size)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_size, self.d_model)
            .with_bias(false)
            .init(device);

        FeedForward { swiglu, w2 }
    }
}

/// Feed-forward transformation network.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    // Swish gated linear unit with trainable parameters.
    swiglu: SwiGlu<B>,
    /// Outer linear.
    w2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(
        &self,
        input: Rank3Tensor<BS, N, S, B>,
    ) -> Result<Rank3Tensor<BS, N, S, B>, crate::Error> {
        Ok(self
            .w2
            .forward(self.swiglu.forward(input.into_inner()))
            .try_into()?)
    }
}

/// Key-value cache for autoregressive models.
pub struct KeyValueCache<B: Backend> {
    key: AutoregressiveCache<B>,
    value: AutoregressiveCache<B>,
}

type KVCacheTensor<B> = Rank4Tensor<BS, K, N, H, B>;
impl<B: Backend> KeyValueCache<B> {
    /// Create a new [key-value cache](KeyValueCache).
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(max_batch_size, num_heads, max_seq_len, d_model, device),
            value: AutoregressiveCache::new(
                max_batch_size,
                num_heads,
                max_seq_len,
                d_model,
                device,
            ),
        }
    }

    /// Computes the complete keys and values.
    pub fn forward(
        &mut self,
        key: KVCacheTensor<B>,
        value: KVCacheTensor<B>,
    ) -> Result<(KVCacheTensor<B>, KVCacheTensor<B>), crate::Error> {
        let k = self.key.forward(key);
        let v = self.value.forward(value);
        Ok((k, v))
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        // We can assume key and value have the same length
        self.key.len()
    }

    /// Reset key-value cache.
    /// Use between different contexts (i.e., for each new prompt).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }
}

/// Configuration to create a [multi-head attention](MultiHeadAttention) module.
#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multi-head attention](MultiHeadAttention) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MultiHeadAttention<B> {
        let head_dim = self.d_model / self.n_heads;

        let wq = LinearConfig::new(self.d_model, self.n_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wk = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wv = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * head_dim, self.d_model)
            .with_bias(false)
            .init(device);

        MultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection.
    wq: Linear<B>,
    /// Key projection.
    wk: Linear<B>,
    /// Value projection.
    wv: Linear<B>,
    /// Output projection.
    wo: Linear<B>,

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward(
        &self,
        input: Rank3Tensor<BS, N, S, B>,
        cache: &mut KeyValueCache<B>,
        rope: &RotaryEncoding<B>,
    ) -> Result<Rank3Tensor<BS, N, S, B>, crate::Error> {
        let device = input.device();
        let [_batch_size, seq_len, _hidden_size] = input.dims();

        let q: Rank3Tensor<BS, N, Q, B> = self.wq.forward(input.clone().into_inner()).try_into()?;
        let k: Rank3Tensor<BS, N, KV, B> =
            self.wk.forward(input.clone().into_inner()).try_into()?;
        let v: Rank3Tensor<BS, N, KV, B> = self.wv.forward(input.into_inner()).try_into()?;

        // [batch_size, num_heads, seq_len, head_dim]
        let q = transpose!(reshape!(q, [BS, { seq_len as i32 } => N, A, H]), U1:U2);
        let k = transpose!(reshape!(k, [BS, { seq_len as i32 } => N, K, H]), U1:U2);
        let v = transpose!(reshape!(v, [BS, { seq_len as i32 } => N, K, H]), U1:U2);

        let cache_seq_len = cache.len();

        let q: Rank4Tensor<BS, A, N, H, B> =
            rope.apply(q.into_inner(), cache_seq_len).try_into()?;
        let k = rope.apply(k.into_inner(), cache_seq_len);

        // Key-value caching
        let (k, v) = cache.forward(k.try_into()?, v)?;

        // Repeat key/value heads if num_kv_heads < num_heads
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Attention scores
        let mut scores = matmul!(q, transpose!(k, U2:U3)) / (self.head_dim as f32).sqrt();

        // Matrix of scores is of size [seqlen, cache_len + seqlen], and the only masked entries are
        // (i, j) for j > cache_len + i, since row i corresponds to token cache_len + i.
        // NOTE: we could possibly improve the mask generation by caching masks for different sequence lengths,
        // though it is probably not necessary at this time.
        if seq_len > 1 {
            let cache_seq_len = cache.len();
            let mask = tril_mask!((cache_seq_len - seq_len) as i64, &device, B, [{ seq_len } => N, { cache_seq_len } => N]);
            let mask = expand!(mask, &scores);
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }

        let scores = softmax!(scores, U3);

        // Output [batch_size, num_heads, seq_len, head_dim]
        let output = matmul!(scores, v);
        let output = transpose!(output, U1:U2);
        let output = reshape!(output, [BS, { seq_len as i32 } => N, S]);
        Ok(self.wo.forward(output.into_inner()).try_into()?)
    }

    /// Repeats a key or value tensor for grouped query attention.
    fn repeat_kv(&self, x: Rank4Tensor<BS, K, N, H, B>) -> Rank4Tensor<BS, A, N, H, B> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            // # attn heads == kv heads
            x.into_inner().try_into().unwrap()
        } else {
            let [_batch_size, _num_kv_heads, seq_len, _head_dim] = x.dims();

            let x = unsqueeze!(x, U2);
            let x = expand!(x, [BS, K, R, { seq_len as i32 } => N, H]);
            reshape!(x, [BS, A, { seq_len as i32 } => N, H])
        }
    }
}



================================================
FILE: examples/burn-llama/src/bin/chat.rs
================================================
#![recursion_limit = "256"]

use std::time::Instant;

use burn::tensor::{backend::Backend, Device};
use burn_llama::{
    llama::{Llama, LlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};
use clap::Parser;

const DEFAULT_PROMPT: &str = "GPU go brrr";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = 128)]
    max_seq_len: usize,

    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, short = 'n', default_value_t = 65)]
    sample_len: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The input prompt.
    #[arg(short, long, default_value_t = String::from(DEFAULT_PROMPT))]
    prompt: String,
}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) {
    let now = Instant::now();
    let generated = llama
        .generate(prompt, sample_len, temperature, sampler)
        .unwrap();
    let elapsed = now.elapsed().as_secs();

    println!("> {}\n", generated.text);
    println!(
        "{} tokens generated ({:.4} tokens/s)\n",
        generated.tokens,
        generated.tokens as f64 / generated.time
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
}

pub fn chat<B: Backend>(args: Config, device: Device<B>) {
    let prompt = args.prompt;

    // Sampling strategy
    let mut sampler = if args.temperature > 0.0 {
        Sampler::TopP(TopP::new(args.top_p, args.seed))
    } else {
        Sampler::Argmax
    };

    let mut llama = LlamaConfig::llama3_2_pretrained::<B>(args.max_seq_len, &device).unwrap();
    println!("Processing prompt: {}", prompt);

    generate(
        &mut llama,
        &prompt,
        args.sample_len,
        args.temperature,
        &mut sampler,
    );
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run(args: Config) {
        let device = WgpuDevice::default();

        chat::<Wgpu>(args, device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::{
        backend::{cuda::CudaDevice, Cuda},
        tensor::f16,
    };

    pub fn run(args: Config) {
        let device = CudaDevice::default();

        chat::<Cuda<f16, i32>>(args, device);
    }
}

pub fn main() {
    // Parse arguments
    let args = Config::parse();

    #[cfg(feature = "wgpu")]
    wgpu::run(args);
    #[cfg(feature = "cuda")]
    cuda::run(args);

    #[cfg(all(not(feature = "wgpu"), not(feature = "cuda")))]
    println!("No backend enabled.");
}



================================================
FILE: examples/burn-llama/src/tokenizer/base.rs
================================================
pub trait Tokenizer {
    /// Load the tokenizer from the provided path.
    fn new(tokenizer_path: &str) -> Result<Self, String>
    where
        Self: Sized;

    /// Encode a string into a list of token identifiers.
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32>;

    /// Decode a list of token identifiers into a string.
    fn decode(&self, tokens: Vec<u32>) -> String;

    /// Beginning of sentence token.
    fn bos(&self) -> String {
        self.decode(vec![self.bos_id()])
    }

    /// Beginning of sentence token identifier.
    fn bos_id(&self) -> u32;

    /// End of sentence token.
    fn eos(&self) -> String {
        self.decode(vec![self.eos_id()])
    }

    /// End of sentence token identifier.
    fn eos_id(&self) -> u32;

    /// Stop token identifiers.
    fn stop_ids(&self) -> Vec<u32>;
}



================================================
FILE: examples/burn-llama/src/tokenizer/mod.rs
================================================
pub mod base;
pub use base::*;

pub mod tiktoken;
pub use tiktoken::*;



================================================
FILE: examples/burn-llama/src/tokenizer/tiktoken.rs
================================================
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHashMap as HashMap;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "<|end_of_text|>";
const EOT_TOKEN: &str = "<|eot_id|>";
const EOM_TOKEN: &str = "<|eom_id|>";

const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;
const SPECIAL_TOKENS: [&str; 11] = [
    BOS_TOKEN,
    EOS_TOKEN,
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    EOM_TOKEN, // end of message
    EOT_TOKEN, // end of turn
    "<|python_tag|>",
];
const PATTERN: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

#[derive(Debug, Clone)]
pub struct Tiktoken {
    bpe: CoreBPE,
    bos_token_id: usize,
    eos_token_id: usize,
    eot_token_id: usize,
    eom_token_id: usize,
}

impl Tokenizer for Tiktoken {
    /// Load the [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    fn new(tiktoken_bpe_file: &str) -> Result<Self, String> {
        let file = File::open(tiktoken_bpe_file).map_err(|e| e.to_string())?;
        let mut mergeable_ranks: HashMap<Vec<u8>, usize> = HashMap::default();

        for line in BufReader::new(file).lines().flatten() {
            let mut parts = line.split(' ');
            let token = STANDARD
                .decode(parts.next().ok_or("Missing token")?)
                .map_err(|e| e.to_string())?;
            let rank = parts
                .next()
                .ok_or("Missing rank")?
                .parse::<usize>()
                .map_err(|e| e.to_string())?;

            mergeable_ranks.insert(token, rank);
        }
        let num_base_tokens = mergeable_ranks.len();

        let special_tokens = [
            SPECIAL_TOKENS
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            (0..NUM_RESERVED_SPECIAL_TOKENS - SPECIAL_TOKENS.len())
                .into_iter()
                .map(|i| format!("<|reserved_special_token_{}|>", i + 2))
                .collect::<Vec<_>>(),
        ]
        .concat();
        let special_tokens = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i + num_base_tokens))
            .collect::<HashMap<String, usize>>();

        let bos_token_id = special_tokens[BOS_TOKEN];
        let eos_token_id = special_tokens[EOS_TOKEN];
        let eot_token_id = special_tokens[EOT_TOKEN];
        let eom_token_id = special_tokens[EOM_TOKEN];

        let bpe =
            CoreBPE::new(mergeable_ranks, special_tokens, PATTERN).map_err(|e| e.to_string())?;
        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            eom_token_id,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let bos_token = if bos { vec![self.bos_token_id] } else { vec![] };
        let eos_token = if eos { vec![self.eos_token_id] } else { vec![] };

        let tokens = self.bpe.encode_with_special_tokens(text);

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .map(|t| t as u32)
            .collect()
    }

    fn decode(&self, tokens: Vec<u32>) -> String {
        self.bpe
            .decode(tokens.into_iter().map(|t| t as usize).collect())
            .expect("Should decode tokens")
    }

    fn bos_id(&self) -> u32 {
        self.bos_token_id as u32
    }

    fn eos_id(&self) -> u32 {
        self.eos_token_id as u32
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![
            self.eos_id(),
            self.eom_token_id as u32,
            self.eot_token_id as u32,
        ]
    }
}



================================================
FILE: examples/candle-llama/README.md
================================================
## candle-llama

This example implements Meta's Llama 3.2 architecture using the [candle](https://github.com/huggingface/candle) framework, leveraging glowstick where possible for compile-time tensor shapes. It was largely copied from the corresponding [candle llama example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama).

Use the following command to test using SmolLM2 135M:

`cargo run --release`

Note that most of the typed shape usage can be found in the model implementation (`src/llama.rs`).




================================================
FILE: examples/candle-llama/Cargo.toml
================================================
[package]
name = "candle-llama"
edition = "2021"

[dependencies]
candle = { version = "0.9", package = "candle-core" }
candle-nn = "0.9"
candle-flash-attn = { version = "0.9", optional = true }
candle-transformers = "0.9"
glowstick.workspace = true
glowstick-candle.workspace = true

accelerate-src = { version = "0.3.2", optional = true }
anyhow = { version = "1", features = ["backtrace"] }
clap = { version = "4.5", features = ["derive"] }
cudarc = { version = "0.16.3", features = ["std", "cublas", "cublaslt", "curand", "driver", "nvrtc", "f16", "cuda-version-from-build-system", "dynamic-linking"], default-features=false, optional = true }
half = { version = "2.5.0", features = ["num-traits", "use-intrinsics", "rand_distr"], optional = true }
hf-hub = { version = "0.4.1", features = ["tokio"] }
intel-mkl-src = { version = "0.8.1", features = ["mkl-static-lp64-iomp"], optional = true }
serde = { version = "1.0.171", features = ["derive"] }
serde_json = { version = "1.0.99" }
thiserror = { version = "2.0.12" }
tokenizers = { version = "0.21.0", default-features = false, features = ["onig"] }
tracing = "0.1.37"
tracing-chrome = "0.7.1"
tracing-subscriber = "0.3.7"

[build-dependencies]
bindgen_cuda = { version = "0.1.1", optional = true }

[features]
default = ["small"]
small = []
smaller = []
smallest = []
accelerate = ["dep:accelerate-src", "candle/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
cuda = ["candle/cuda", "candle-nn/cuda", "candle-transformers/cuda", "dep:bindgen_cuda"]
cudnn = ["candle/cudnn", "candle-nn/cudnn", "candle-transformers/cudnn"]
flash-attn = ["dep:candle-flash-attn", "candle-transformers/flash-attn"]
mkl = ["dep:intel-mkl-src", "candle/mkl", "candle-nn/mkl", "candle-transformers/mkl"]
metal = ["candle/metal", "candle-nn/metal"]



================================================
FILE: examples/candle-llama/src/llama.rs
================================================
//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use candle::{DType, Device, IndexOp, Tensor as CandleTensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use glowstick::{
    num::{U0, U1, U2, U3, U32, U8192},
    Shape2, Shape3, Shape4,
};
use glowstick_candle::{cat, expand, matmul, narrow, reshape, softmax, transpose, Error, Tensor};
use std::{collections::HashMap, f32::consts::PI};

use crate::shape::{A, B, C, H, K, KV, N, Q, S};

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

type CacheTensor = Tensor<Shape4<B, K, N, H>>;
#[derive(Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor<Shape2<N, N>>>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(CacheTensor, CacheTensor)>>,
    cos: Tensor<Shape2<U8192, U32>>,
    sin: Tensor<Shape2<U8192, U32>>,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
    ) -> Result<Self, Error> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = CandleTensor::new(theta, device)?;

        let idx_theta = CandleTensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos: Tensor<Shape2<U8192, U32>> = idx_theta.cos()?.to_dtype(dtype)?.try_into()?;
        let sin: Tensor<Shape2<U8192, U32>> = idx_theta.sin()?.to_dtype(dtype)?.try_into()?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor<Shape2<N, N>>, Error> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask: Tensor<Shape2<N, N>> =
                CandleTensor::from_slice(&mask, (t, t), &self.device)?.try_into()?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor<Shape4<B, N, A, H>>,
    k: &Tensor<Shape4<B, N, A, H>>,
    v: &Tensor<Shape4<B, N, A, H>>,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor<Shape4<B, N, A, H>>, Error> {
    candle_flash_attn::flash_attn(q.inner(), k.inner(), v.inner(), softmax_scale, causal)?
        .try_into()
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(
    _: &Tensor<Shape4<B, N, A, H>>,
    _: &Tensor<Shape4<B, N, A, H>>,
    _: &Tensor<Shape4<B, N, A, H>>,
    _: f32,
    _: bool,
) -> Result<Tensor<Shape4<B, N, A, H>>, Error> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_attn_head_rotary_emb(
        &self,
        x: &Tensor<Shape4<B, A, N, H>>,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor<Shape4<B, A, N, H>>, Error> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.inner().dims4()?;
        let cos = narrow!(&cache.cos, U0: [{ index_pos }, { seq_len }] => N)?;
        let sin = narrow!(&cache.sin, U0: [{ index_pos }, { seq_len }] => N)?;
        candle_nn::rotary_emb::rope(x.inner(), cos.inner(), sin.inner())?.try_into()
    }

    fn apply_kv_head_rotary_emb(
        &self,
        x: &Tensor<Shape4<B, K, N, H>>,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor<Shape4<B, K, N, H>>, Error> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.inner().dims4()?;
        let cos = narrow!(&cache.cos, U0: [{ index_pos }, { seq_len }] => N)?;
        let sin = narrow!(&cache.sin, U0: [{ index_pos }, { seq_len }] => N)?;
        candle_nn::rotary_emb::rope(x.inner(), cos.inner(), sin.inner())?.try_into()
    }

    fn forward(
        &self,
        x: &Tensor<Shape3<B, N, S>>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor<Shape3<B, N, S>>, Error> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _hidden_size) = x.inner().dims3()?;
        let q: Tensor<Shape3<B, N, Q>> = self.q_proj.forward(x.inner())?.try_into()?;
        let k: Tensor<Shape3<B, N, KV>> = self.k_proj.forward(x.inner())?.try_into()?;
        let v: Tensor<Shape3<B, N, KV>> = self.v_proj.forward(x.inner())?.try_into()?;

        let q =
            transpose!(reshape!(&q, [() => B, { seq_len } => N, A, H])?, U1:U2)?.contiguous()?;
        let k =
            transpose!(reshape!(&k, [() => B, { seq_len } => N, K, H])?, U1:U2)?.contiguous()?;
        let mut v =
            transpose!(reshape!(&v, [() => B, { seq_len } => N, K, H])?, U1:U2)?.contiguous()?;

        let q = self.apply_attn_head_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_kv_head_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = cat!(vec![cache_k, &k].as_slice(), U2 => N)?.contiguous()?;
                v = cat!(vec![cache_v, &v].as_slice(), U2 => N)?.contiguous()?;
                let k_seq_len = k.inner().dims()[2];
                if k_seq_len > self.max_position_embeddings {
                    k = narrow!(
                        &k,
                        U2: [{ k_seq_len - self.max_position_embeddings }, self.max_position_embeddings] => N
                    )?;
                }
                let v_seq_len = v.inner().dims()[2];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = narrow!(
                        &v,
                        U2: [{ v_seq_len - self.max_position_embeddings }, self.max_position_embeddings] => N
                    )?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = transpose!(q, U1:U2)?;
            let k = transpose!(k, U1:U2)?;
            let v = transpose!(v, U1:U2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            transpose!(flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?, U1:U2)
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (matmul!(q, transpose!(k, U2:U3)?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = expand!(&cache.mask(seq_len)?, &att)?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };

            let att = softmax!(att, U3)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            matmul!(att, &v.contiguous()?)?.to_dtype(in_dtype)
        }?;
        let y = transpose!(y, U1:U2)?;
        let y = reshape!(y, [{ b_sz } => B, { seq_len } => N, S])?;
        let y = self.o_proj.forward(y.inner())?;
        y.try_into()
    }

    fn repeat_kv(
        &self,
        x: Tensor<Shape4<B, K, N, H>>,
    ) -> Result<Tensor<Shape4<B, A, N, H>>, Error> {
        candle_transformers::utils::repeat_kv(
            x.into_inner(),
            self.num_attention_heads / self.num_key_value_heads,
        )?
        .try_into()
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

fn masked_fill(
    on_false: &Tensor<Shape4<B, A, N, N>>,
    mask: &Tensor<Shape4<B, A, N, N>>,
    on_true: f32,
) -> Result<Tensor<Shape4<B, A, N, N>>, Error> {
    let shape = mask.inner().shape();
    let on_true =
        CandleTensor::new(on_true, on_false.inner().device())?.broadcast_as(shape.dims())?;
    let m = mask
        .inner()
        .where_cond(&on_true, on_false.inner())?
        .try_into()?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor<Shape3<B, N, S>>) -> Result<Tensor<Shape3<B, N, S>>, Error> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x.inner())?)?
            * self.c_fc2.forward(x.inner())?)?;
        self.c_proj.forward(&x)?.try_into()
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor<Shape3<B, N, S>>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor<Shape3<B, N, S>>, Error> {
        let _enter = self.span.enter();
        let residual = x;
        let x: Tensor<Shape3<B, N, S>> = self.rms_1.forward(x.inner())?.try_into()?;
        let x = (&self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let x = (&self
            .mlp
            .forward(&self.rms_2.forward(x.inner())?.try_into()?)?
            + x)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    pub eos_tokens: Option<LlamaEosToks>,
}

impl Llama {
    pub fn forward(
        &self,
        x: &Tensor<Shape2<B, N>>,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor<Shape2<B, C>>, Error> {
        let (_b_sz, seq_len) = x.inner().dims2()?;
        let mut x: Tensor<Shape3<B, N, S>> = self.wte.forward(x.inner())?.try_into()?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(x.inner())?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?.try_into()?;
        Ok(logits)
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self, candle::Error> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            eos_tokens: cfg.eos_token_id.clone(),
        })
    }
}



================================================
FILE: examples/candle-llama/src/main.rs
================================================
use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use glowstick::{
    num::{U0, U1},
    Shape2,
};
use glowstick_candle::tensor::Tensor;
use glowstick_candle::{cat, narrow, squeeze};
use hf_hub::{api::sync::Api, Repo, RepoType};
use llama::{Llama, LlamaConfig, LlamaEosToks};
use tokenizers::Tokenizer;

mod llama;
mod shape;

use shape::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Rank mismatch: runtime ({runtime}) vs type-level ({type_level})")]
    RankMismatch { runtime: usize, type_level: usize },

    #[error("Dimension mismatch: expected {type_level} for dim {dim} but received {runtime}")]
    DimensionMismatch {
        dim: usize,
        runtime: usize,
        type_level: usize,
    },

    #[error("Couldn't find the EOS token.")]
    MissingEosToken,

    #[error("No token streams!")]
    NoTokenStreams,

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("Encode error: {0}")]
    Encode(String),

    #[error("{0}")]
    Candle(#[from] candle::Error),

    #[error("glowstick error: {0}")]
    Glowstick(#[from] glowstick_candle::Error),
}

enum Model {
    Smol(llama::Llama, llama::Cache),
}

impl Model {
    fn forward(
        &mut self,
        xs: &Tensor<Shape2<B, N>>,
        s: usize,
    ) -> Result<Tensor<Shape2<B, C>>, Error> {
        match self {
            Self::Smol(ref mut model, ref mut cache) => Ok(model.forward(xs, s, cache)?),
        }
    }

    fn eos_tokens(&self, t: &TokenOutputStream<'_>) -> Result<Vec<u32>, Error> {
        match self {
            Self::Smol(m, _) => {
                let eos = t.get_token("</s>").ok_or(Error::MissingEosToken);
                match &m.eos_tokens {
                    Some(LlamaEosToks::Multiple(v)) => {
                        Ok(v.iter().copied().chain(eos.ok()).collect())
                    }
                    Some(LlamaEosToks::Single(n)) => {
                        Ok(std::iter::once(*n).chain(eos.ok()).collect())
                    }
                    None => Ok(std::iter::once(eos?).collect()),
                }
            }
        }
    }
}

struct TextGeneration<'a> {
    model: Model,
    device: Device,
    token_streams: Vec<TokenOutputStream<'a>>,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<'a> TextGeneration<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: &'a Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        num_return_sequences: usize,
        device: &Device,
    ) -> Self {
        let temperature = temp.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            },
        };
        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);
        Self {
            model,
            token_streams: (0..num_return_sequences)
                .map(|_| TokenOutputStream::new(tokenizer))
                .collect::<Vec<_>>(),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<(), Error> {
        use std::io::Write;

        let n_outputs = self.token_streams.len();
        let eos_tokens = self
            .token_streams
            .first()
            .map(|t| self.model.eos_tokens(t))
            .ok_or(Error::NoTokenStreams)??;
        self.token_streams.iter_mut().for_each(|tokenizer| {
            tokenizer.clear();
        });

        enum TokenList {
            Generating(Vec<u32>),
            Terminated(Vec<u32>),
        }
        impl TokenList {
            fn len(&self) -> usize {
                match self {
                    Self::Generating(v) => v.len(),
                    Self::Terminated(v) => v.len(),
                }
            }

            fn iter(&self) -> impl Iterator<Item = &u32> {
                match self {
                    Self::Generating(v) => v.iter(),
                    Self::Terminated(v) => v.iter(),
                }
            }

            fn ctxt(&self, start_pos: usize) -> &[u32] {
                match self {
                    Self::Generating(v) => &v[start_pos..],
                    Self::Terminated(v) => &v[start_pos..],
                }
            }

            fn push(&mut self, t: u32) {
                match self {
                    Self::Generating(v) => {
                        v.push(t);
                    }
                    Self::Terminated(_v) => {}
                }
            }

            fn terminate(&mut self) {
                let v = match self {
                    Self::Generating(v) | Self::Terminated(v) => std::mem::take(v),
                };
                *self = Self::Terminated(v);
            }

            fn is_terminated(&self) -> bool {
                matches!(self, Self::Terminated { .. })
            }
        }

        let mut token_lists = self
            .token_streams
            .iter()
            .map(|t| {
                Ok::<_, Error>(TokenList::Generating(
                    t.tokenizer()
                        .encode(prompt, true)
                        .map_err(|e| Error::Encode(e.to_string()))?
                        .get_ids()
                        .to_vec(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut seq_len = 0;
        let mut finished_sequences = vec![vec![]; token_lists.len()];
        for (i, (v, st)) in token_lists.iter().zip(&mut self.token_streams).enumerate() {
            seq_len = v.len();
            for &t in v.iter() {
                if let Some(t) = st.next_token(t)? {
                    if n_outputs == 1 {
                        print!("{t}")
                    }
                    finished_sequences[i].push(t);
                }
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { seq_len };
            let Some(generating) = token_lists
                .iter()
                .find(|v| matches!(v, TokenList::Generating(_)))
            else {
                break;
            };
            let start_pos = generating.len().saturating_sub(context_size);
            let inputs = cat!(token_lists.iter().map(|v| {
                let start_pos = v.len().saturating_sub(context_size);
                let ctxt = v.ctxt(start_pos);
                let input: Tensor<Shape2<U1, N>> = candle::Tensor::new(ctxt, &self.device)?.unsqueeze(0)?.try_into()?;

                Ok::<_, Error>(input)
            }).collect::<Result<Vec<Tensor<_>>, _>>()?.as_slice(), U0 => B)?;

            let logits = self.model.forward(&inputs, start_pos)?;
            for (i, (v, st)) in token_lists
                .iter_mut()
                .zip(&mut self.token_streams)
                .enumerate()
            {
                let logits = narrow!(&logits, U0: [{ i }, U1])?;
                let logits = squeeze![&logits, U0]?.to_dtype(DType::F32)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = v.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        logits.inner(),
                        self.repeat_penalty,
                        v.ctxt(start_at),
                    )?
                    .try_into()?
                };

                let next_token = self.logits_processor.sample(logits.inner())?;
                v.push(next_token);
                generated_tokens += 1;
                if eos_tokens.contains(&next_token) && matches!(v, TokenList::Generating(_)) {
                    v.terminate();
                } else if let Some(t) = st.next_token(next_token)? {
                    if n_outputs == 1 {
                        print!("{t}");
                    } else if generated_tokens % 100 == 0 {
                        println!("Generated {} tokens", generated_tokens);
                    }
                    finished_sequences[i].push(t);
                }
            }
            if token_lists.iter().all(|l| l.is_terminated()) {
                break;
            }
        }
        let dt = start_gen.elapsed();

        for (i, (st, finished)) in self
            .token_streams
            .iter()
            .zip(&mut finished_sequences)
            .enumerate()
        {
            if let Some(rest) = st.decode_rest()? {
                finished.push(rest);
            }
            if n_outputs > 1 {
                println!("[OUTPUT SEQUENCE {}]", i + 1);
                for t in finished.iter() {
                    print!("{t}");
                }
            }
            println!("\n");
            std::io::stdout().flush()?;
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "smol-135m")]
    S135m,
    #[value(name = "smol-360m")]
    S360m,
    #[value(name = "smol-1.7b")]
    S1_7b,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[arg(long, default_value = "GPU go brrr")]
    prompt: String,

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    #[arg(long, default_value = "CPU go brrr")]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 512)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(short, long, default_value_t = 1)]
    num_return_sequences: usize,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;

    #[cfg(feature = "small")]
    let model = WhichModel::S1_7b;
    #[cfg(feature = "smaller")]
    let model = WhichModel::S360m;
    #[cfg(feature = "smallest")]
    let model = WhichModel::S135m;

    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            let (version, size) = match model {
                WhichModel::S135m => ("2", "135M"),
                WhichModel::S360m => ("2", "360M"),
                WhichModel::S1_7b => ("2", "1.7B"),
            };
            format!("HuggingFaceTB/SmolLM{version}-{size}")
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => match model {
            WhichModel::S135m | WhichModel::S360m | WhichModel::S1_7b => {
                vec![repo.get("model.safetensors")?]
            }
        },
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config_file = repo.get("config.json")?;
    let device = device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = match model {
        WhichModel::S135m | WhichModel::S360m | WhichModel::S1_7b => {
            let config = serde_json::from_slice::<LlamaConfig>(&std::fs::read(&config_file)?)?
                .into_config(args.use_flash_attn);

            let cache = llama::Cache::new(true, dtype, &config, &device)?;
            Model::Smol(Llama::load(vb, &config)?, cache)
        }
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        &tokenizer,
        args.seed,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.num_return_sequences,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> TokenOutputStream<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> &'a tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> candle::Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>, Error> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>, Error> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> candle::Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(safetensors_files)
}



================================================
FILE: examples/candle-llama/src/shape.rs
================================================
use std::ops::{Add, Div, Mul};

use glowstick::dyndims;
use glowstick::num::{U1000, U152, U49};

dyndims! {
    B: BatchSize,
    N: SequenceLength
}

type U49152 = <<U49 as Mul<U1000>>::Output as Add<U152>>::Output;
pub type C = U49152; // Vocabulary
pub type H = <S as Div<A>>::Output; // Head-dim
pub type Q = S;
pub type KV = <<S as Div<A>>::Output as Mul<K>>::Output;

#[cfg(all(not(feature = "smaller"), not(feature = "smallest")))]
mod config_dims {
    use glowstick::num::{U2048, U32};

    pub type A = U32; // Attention Heads
    pub type K = U32; // Key-Value Heads
    pub type S = U2048; // Hidden Size
}

#[cfg(all(feature = "smaller", not(feature = "smallest")))]
mod config_dims {
    use glowstick::num::{U15, U5, U960};

    pub type A = U15; // Attention Heads
    pub type K = U5; // Key-Value Heads
    pub type S = U960; // Hidden Size
}

#[cfg(feature = "smallest")]
mod config_dims {
    use glowstick::num::{U3, U576, U9};

    pub type A = U9; // Attention Heads
    pub type K = U3; // Key-Value Heads
    pub type S = U576; // Hidden Size
}

pub use config_dims::*;



================================================
FILE: glowstick-burn/Cargo.toml
================================================
[package]
name = "glowstick-burn"
description = "Integration of glowstick with the burn tensor"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
categories.workspace = true
keywords.workspace = true

[dependencies]
burn = { version = "0.17", default-features = false }
glowstick.workspace = true
thiserror.workspace = true

[dev-dependencies]
burn = { version = "0.17", default-features = false, features = ["ndarray"] }




================================================
FILE: glowstick-burn/src/lib.rs
================================================
pub mod op;
pub mod tensor;

pub use tensor::{Error, Tensor};



================================================
FILE: glowstick-burn/src/tensor.rs
================================================
use std::marker::PhantomData;
use std::ops::Range;

use burn::tensor::{
    BasicOps, Bool, DType, ElementConversion, Int, Numeric, Tensor as BTensor, TensorData,
};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::Equal;
use glowstick::{
    num::{U0, U1},
    Dimension, Dimensioned, Shape, TensorShape,
};
use glowstick::{Arrayify, IsFragEqual, ShapeFragment};

pub const fn rank<B: Backend, const N: usize>(_t: &BTensor<B, N>) -> usize {
    N
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "Rank mismatch: the const generic rank provided to burn does not match the type-level rank associated with the glowstick shape. Const: ({const_level}) Type-level: ({type_level})"
    )]
    RankMismatch {
        const_level: usize,
        type_level: usize,
    },

    #[error("Dimension mismatch: expected {type_level} for dim {dim} but received {runtime}")]
    DimensionMismatch {
        dim: usize,
        runtime: usize,
        type_level: usize,
    },
}

pub struct Tensor<T, S: Shape>(pub(crate) T, pub(crate) PhantomData<S>);
impl<T: Clone, S: Shape> Clone for Tensor<T, S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<B, S, Dtype, const D: usize> glowstick::Tensor for Tensor<BTensor<B, D, Dtype>, S>
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B>,
{
    type Shape = S;
}

impl<B, S, Dtype, const D: usize> TryFrom<BTensor<B, D, Dtype>> for Tensor<BTensor<B, D, Dtype>, S>
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B> + BasicOps<B>,
{
    type Error = Error;
    fn try_from(x: BTensor<B, D, Dtype>) -> Result<Self, Self::Error> {
        if S::RANK != D {
            return Err(Error::RankMismatch {
                const_level: D,
                type_level: S::RANK,
            });
        }

        for (dim, (a, b)) in x.dims().iter().copied().zip(S::iter()).enumerate() {
            if a != b {
                return Err(Error::DimensionMismatch {
                    dim,
                    runtime: a,
                    type_level: b,
                });
            }
        }

        Ok(Self(x, PhantomData))
    }
}

impl<B, S1, S2, const N: usize> std::ops::Add<Tensor<BTensor<B, N>, TensorShape<S1>>>
    for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S1: ShapeFragment,
    S2: ShapeFragment,
    (S1, S2): IsFragEqual,
{
    type Output = Self;
    fn add(self, rhs: Tensor<BTensor<B, N>, TensorShape<S1>>) -> Self {
        Tensor(self.0 + rhs.0, PhantomData)
    }
}

impl<B, S, D, const N: usize> std::ops::Add<i32> for Tensor<BTensor<B, N, D>, TensorShape<S>>
where
    B: Backend,
    D: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S: ShapeFragment,
{
    type Output = Self;
    fn add(self, rhs: i32) -> Self {
        Tensor(self.0 + rhs, PhantomData)
    }
}

impl<B, S1, S2, const N: usize> std::ops::Sub<Tensor<BTensor<B, N>, TensorShape<S1>>>
    for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S1: ShapeFragment,
    S2: ShapeFragment,
    (S1, S2): IsFragEqual,
{
    type Output = Self;
    fn sub(self, rhs: Tensor<BTensor<B, N>, TensorShape<S1>>) -> Self {
        Tensor(self.0 - rhs.0, PhantomData)
    }
}

impl<B, S1, S2, const N: usize> std::ops::Div<Tensor<BTensor<B, N>, TensorShape<S1>>>
    for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S1: ShapeFragment,
    S2: ShapeFragment,
    (S1, S2): IsFragEqual,
{
    type Output = Self;
    fn div(self, rhs: Tensor<BTensor<B, N>, TensorShape<S1>>) -> Self {
        Tensor(self.0 / rhs.0, PhantomData)
    }
}
impl<B, S2, const N: usize> std::ops::Div<f32> for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S2: ShapeFragment,
{
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Tensor(self.0.div_scalar(rhs), PhantomData)
    }
}
impl<B, S2, const N: usize> std::ops::Div<f64> for Tensor<BTensor<B, N>, TensorShape<S2>>
where
    B: Backend,
    S2: ShapeFragment,
{
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Tensor(self.0.div_scalar(rhs), PhantomData)
    }
}

impl<B, S, const D: usize> std::ops::Mul<f64> for Tensor<BTensor<B, D>, S>
where
    B: Backend,
    S: Shape,
{
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs, PhantomData)
    }
}

impl<B, S, Dtype, const D: usize> Tensor<BTensor<B, D, Dtype>, S>
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B>,
{
    pub const fn rank() -> usize {
        S::RANK
    }

    pub fn into_inner(self) -> BTensor<B, D, Dtype> {
        self.0
    }

    pub fn inner(&self) -> &BTensor<B, D, Dtype> {
        &self.0
    }
}

impl<B, S, Dtype> Tensor<BTensor<B, 1, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B>,
{
    fn check_dim(inner: &BTensor<B, 1, Dtype>) -> Result<(), Error> {
        if inner.dims()[0] != <S::Dim<U0> as Dimension>::USIZE {
            return Err(Error::DimensionMismatch {
                dim: 0,
                runtime: inner.dims()[0],
                type_level: <S::Dim<U0> as Dimension>::USIZE,
            });
        }

        Ok(())
    }
}

impl<B, S> Tensor<BTensor<B, 1, Int>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
{
    pub fn arange(range: Range<i64>, device: &B::Device) -> Self {
        Self(BTensor::<B, 1, Int>::arange(range, device), PhantomData)
    }

    pub fn float(self) -> Tensor<BTensor<B, 1>, S> {
        Tensor(self.0.float(), PhantomData)
    }
}

impl<B, S> Tensor<BTensor<B, 1>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
{
    pub fn from_floats(f: &[f32], d: &B::Device) -> Result<Self, Error> {
        Self::check_rank()?;
        let inner: BTensor<B, 1> = BTensor::from_floats(f, d);
        Self::check_dim(&inner)?;

        Ok(Self(inner, PhantomData))
    }
}

impl<B, S, const N: usize> Tensor<BTensor<B, N, Int>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    (<S as Shape>::Rank, U1): Equal,
{
    pub fn from_ints<T: Into<TensorData>>(n: T, d: &B::Device) -> Self {
        Self(BTensor::from_ints(n, d), PhantomData)
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B>,
{
    fn check_rank() -> Result<(), Error> {
        if S::RANK != N {
            return Err(Error::RankMismatch {
                const_level: 1,
                type_level: S::RANK,
            });
        }

        Ok(())
    }
}

impl<B, S, const N: usize> Tensor<BTensor<B, N>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
{
    pub fn cast(self, dtype: DType) -> Tensor<BTensor<B, N>, S> {
        Self(self.0.cast(dtype), PhantomData)
    }

    pub fn cos(self) -> Self {
        Self(self.0.cos(), PhantomData)
    }

    pub fn sin(self) -> Self {
        Self(self.0.cos(), PhantomData)
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B>,
{
    pub fn dims(&self) -> [usize; N] {
        self.0.dims()
    }

    pub fn device(&self) -> <B as Backend>::Device {
        self.0.device()
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B> + Numeric<B>,
{
    pub fn mask_fill<E>(self, mask: Tensor<BTensor<B, N, Bool>, S>, value: E) -> Self
    where
        E: ElementConversion,
    {
        Self(self.0.mask_fill(mask.into_inner(), value), PhantomData)
    }

    pub fn to_data(&self) -> TensorData {
        self.0.to_data()
    }

    pub fn into_data(self) -> TensorData {
        self.0.into_data()
    }
}

impl<B, S, Dtype, const N: usize> Tensor<BTensor<B, N, Dtype>, S>
where
    B: Backend,
    S: Shape,
    <S as Shape>::Fragment: Arrayify<usize, Out = [usize; N]>,
    (S, U0): Dimensioned,
    Dtype: TensorKind<B> + BasicOps<B> + Numeric<B>,
{
    pub fn ones(device: &B::Device) -> Self {
        Self(
            BTensor::ones(
                <<S as Shape>::Fragment as glowstick::Arrayify<usize>>::value(),
                device,
            ),
            PhantomData,
        )
    }

    pub fn zeros(device: &B::Device) -> Self {
        Self(
            BTensor::zeros(
                <<S as Shape>::Fragment as glowstick::Arrayify<usize>>::value(),
                device,
            ),
            PhantomData,
        )
    }

    pub fn from_data<E: burn::tensor::Element>(data: Vec<E>, device: &B::Device) -> Self {
        Self(
            BTensor::from_data(
                TensorData::new(
                    data,
                    <<S as Shape>::Fragment as glowstick::Arrayify<usize>>::value(),
                ),
                device,
            ),
            PhantomData,
        )
    }
}

impl<B, S, const N: usize> Tensor<BTensor<B, N>, S>
where
    B: Backend,
    S: Shape,
{
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt(), PhantomData)
    }
}



================================================
FILE: glowstick-burn/src/op/argmax.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{BasicOps, Int, Numeric, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::Greater;
use glowstick::{
    num::{Unsigned, U0, U1},
    op::narrow,
    Shape,
};

use crate::Tensor;

/// Returns the indices of the maximum values along a specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{argmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let argmaxed = argmax!(a, U1);
///
/// assert_eq!(argmaxed.dims(), [2, 1, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! argmax {
    [$t:expr,$i:ty] => {{
        use $crate::op::argmax::ArgMax;
        ($t, std::marker::PhantomData::<$i>).argmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::argmax![$crate::argmax![$t,$i],$($is),+]
    }};
}

pub trait ArgMax {
    type Out;
    fn argmax(self) -> Self::Out;
}
impl<B, D, S, const N: usize, Dim> ArgMax for (Tensor<BTensor<B, N, D>, S>, PhantomData<Dim>)
where
    B: Backend,
    D: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Tensor<BTensor<B, N, Int>, <(S, Dim, U0, U1) as narrow::Compatible>::Out>;
    fn argmax(self) -> Self::Out {
        Tensor(
            self.0.into_inner().argmax(<Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/cat.rs
================================================
use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{BasicOps, Tensor as BTensor, TensorKind},
};
use glowstick::{num::Unsigned, op::cat_dyn, Shape};

use crate::Tensor;

/// Concatenates the given tensors along a specified dimension.
/// A dynamic dimension must be provided for the return type.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{cat, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let b = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let concatenated = cat!(vec![a, b], U0 => B);
///
/// assert_eq!(concatenated.dims(), [2, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! cat {
    ($ts:expr,$i:ty => $d:ty) => {{
        use $crate::op::cat::Cat;
        (
            $ts,
            std::marker::PhantomData::<$i>,
            std::marker::PhantomData::<$d>,
        )
            .cat()
    }};
}

pub trait Cat {
    type Out;
    fn cat(self) -> Self::Out;
}
impl<B, Dt, S, I, D, const N: usize> Cat
    for (
        Vec<Tensor<BTensor<B, N, Dt>, S>>,
        PhantomData<I>,
        PhantomData<glowstick::Dyn<D>>,
    )
where
    B: Backend,
    Dt: TensorKind<B> + BasicOps<B>,
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Tensor<BTensor<B, N, Dt>, <(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>;
    fn cat(self) -> Self::Out {
        Tensor(
            BTensor::cat(
                self.0.into_iter().map(Tensor::into_inner).collect(),
                <I as Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/expand.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{BasicOps, BroadcastArgs, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{op::broadcast, Shape};

use crate::Tensor;

/// Expands the lefthand tensor to the shape of the provided righthand tensor
/// or shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U1, U2>>::ones(&device);
/// let b = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let c = expand!(a.clone(), &b);
/// let d = expand!(a, [U1, U4, U3, U2]);
///
/// assert_eq!(c.dims(), [1, 4, 3, 2]);
/// assert_eq!(d.dims(), [1, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
///
/// When expanding to a shape, a combination of type-level integers and
/// expressions bound to dynamic dimensions may be provided.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::{U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let [batch_size, seq_len] = [4, 12];
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U1, U2>>::ones(&device);
/// let b = expand!(a, [{ batch_size } => B, { seq_len } => N, U3, U2]);
///
/// assert_eq!(b.dims(), [4, 12, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! expand {
    ($t:expr,[$($ds:tt)+]) => {{
        type S = glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        use $crate::op::expand::Expand;
        (
            $t,
            std::marker::PhantomData::<S>,
        )
            .expand($crate::reshape_val!($($ds)+).into_array())
    }};
    ($t1:expr,$t2:expr) => {{
        use $crate::op::expand::Expand;
        (
            $t1,
            $t2,
        )
            .expand($t2.inner().shape().dims())
    }}
}

pub trait Expand<A, const N: usize, const M: usize>
where
    A: BroadcastArgs<N, M>,
{
    type Out;
    fn expand(self, shape: A) -> Self::Out;
}
impl<B, S1, S2, D1, D2, const N: usize, const M: usize> Expand<[usize; M], N, M>
    for (
        Tensor<BTensor<B, N, D1>, S1>,
        &Tensor<BTensor<B, M, D2>, S2>,
    )
where
    B: Backend,
    S1: Shape,
    S2: Shape,
    D1: TensorKind<B> + BasicOps<B>,
    D2: TensorKind<B>,
    (S2, S1): broadcast::Compatible,
{
    type Out = Tensor<BTensor<B, M, D1>, <(S2, S1) as broadcast::Compatible>::Out>;
    fn expand(self, shape: [usize; M]) -> Self::Out {
        Tensor(self.0.into_inner().expand(shape), PhantomData)
    }
}
impl<B, S1, S2, const N: usize, const M: usize> Expand<[i32; M], N, M>
    for (Tensor<BTensor<B, N>, S1>, PhantomData<S2>)
where
    B: Backend,
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Tensor<BTensor<B, M>, <(S2, S1) as broadcast::Compatible>::Out>;
    fn expand(self, shape: [i32; M]) -> Self::Out {
        Tensor(self.0.into_inner().expand(shape), PhantomData)
    }
}



================================================
FILE: glowstick-burn/src/op/flatten.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{num::Unsigned, op::flatten, Shape};

use crate::Tensor;

/// Flattens the given tensor from the specified start dimension to the end
/// dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{flatten, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let flattened = flatten!(a.clone(), [U0, U2]);
///
/// assert_eq!(flattened.dims(), [12, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! flatten {
    ($t:expr,[$d1:ty,$d2:ty]) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten()
    }};
    ($t:expr,[$d1:ty,$d2:ty],$([$d1s:ty,$d2s:ty]),+) => {{
        use $crate::op::flatten::Flatten;
        let t = (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten();

        $crate::flatten!(&t, $([$d1s,$d2s]),+)
    }};
}

pub trait Flatten<const M: usize> {
    type Out;
    fn flatten(self) -> Self::Out;
}
impl<B, S, Dim1, Dim2, const N: usize, const M: usize> Flatten<M>
    for (
        Tensor<BTensor<B, N>, S>,
        PhantomData<Dim1>,
        PhantomData<Dim2>,
    )
where
    B: Backend,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Tensor<BTensor<B, M>, <(S, Dim1, Dim2) as flatten::Compatible>::Out>;
    fn flatten(self) -> Self::Out {
        Tensor(
            self.0.into_inner().flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/gather.rs
================================================
use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{BasicOps, Int, Numeric, Tensor as BTensor, TensorKind},
};
use glowstick::{num::Unsigned, op::gather, Shape};

use crate::Tensor;

/// Gathers the elements from a tensor at the provided indices along a specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor, Int};
/// use glowstick_burn::{gather, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U1, U1, U4>>::from_data(vec![1., 2., 3., 4.], &device);
/// let b = Tensor::<BurnTensor<Backend, 3, Int>, Shape3<U1, U1, U2>>::from_data(vec![1, 2], &device);
/// let gathered = gather!(a, b, U2);
///
/// assert_eq!(gathered.inner().to_data().to_vec::<f32>().unwrap(), vec![2., 3.]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! gather {
    ($t1:expr,$t2:expr,$d:ty) => {{
        use $crate::op::gather::Gather;
        (
            $t1,
            std::marker::PhantomData,
            $t2,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
        )
            .gather()
    }};
}

pub trait Gather {
    type Out;
    fn gather(self) -> Self::Out;
}
impl<B, D1, S1, S2, Dim, const N: usize> Gather
    for (
        Tensor<BTensor<B, N, D1>, S1>,
        PhantomData<S1>,
        Tensor<BTensor<B, N, Int>, S2>,
        PhantomData<S2>,
        PhantomData<Dim>,
    )
where
    B: Backend,
    D1: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S1: Shape,
    S2: Shape,
    Dim: Unsigned,
    (S1, S2, Dim): gather::Compatible,
{
    type Out = Tensor<BTensor<B, N, D1>, <(S1, S2, Dim) as gather::Compatible>::Out>;
    fn gather(self) -> Self::Out {
        Tensor(
            self.0
                .into_inner()
                .gather(<Dim as Unsigned>::USIZE, self.2.into_inner()),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/log_softmax.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::activation::log_softmax;
use burn::tensor::Tensor as BTensor;

use glowstick::cmp::Greater;
use glowstick::{num::Unsigned, Shape};

use crate::Tensor;

/// Applies the log softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{log_softmax, Tensor};
/// use glowstick::{Shape3, num::*, dyndims};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let logsoftmaxed = log_softmax!(a.clone(), U1);
///
/// assert_eq!(logsoftmaxed.dims(), [2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! log_softmax {
    [$t:expr,$i:ty] => {{
        use $crate::op::log_softmax::LogSoftmax;
        ($t, std::marker::PhantomData::<$i>).log_softmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::log_softmax![$crate::log_softmax![$t,$i],$($is),+]
    }};
}

pub trait LogSoftmax {
    type Out;
    fn log_softmax(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> LogSoftmax for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Tensor<BTensor<B, N>, S>;
    fn log_softmax(self) -> Self::Out {
        Tensor(
            log_softmax(self.0.into_inner(), <Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/matmul.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{op::matmul, Shape, TensorShape};

use crate::Tensor;

/// Performs matrix multiplication of the lefthand tensor and righthand tensor(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{flatten, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U4, U3, U2>>::ones(&device);
/// let flattened = flatten!(a.clone(), [U0, U2]);
///
/// assert_eq!(flattened.dims(), [12, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        $crate::matmul![$crate::matmul!($t1, $t2),$($t2s),+]
    }};
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<B, S1, S2, const N: usize> Matmul for (Tensor<BTensor<B, N>, S1>, Tensor<BTensor<B, N>, S2>)
where
    B: Backend,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Tensor<BTensor<B, N>, TensorShape<<(S1, S2) as matmul::Compatible>::Out>>;
    fn matmul(self) -> Self::Out {
        Tensor(self.0.into_inner().matmul(self.1.into_inner()), PhantomData)
    }
}



================================================
FILE: glowstick-burn/src/op/mean_dim.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{
    num::{Unsigned, U0, U1},
    op::narrow,
    Shape,
};

use crate::Tensor;

/// Computes the mean of a tensor along a specified dimension, resulting in a tensor with size `U1` at that dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{mean_dim, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U2, U3, U4, U5>>::ones(&device);
/// let meaned = mean_dim!(a, U1);
///
/// assert_eq!(meaned.dims(), [2, 1, 4, 5]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! mean_dim {
    [$t:expr,$i:ty] => {{
        use $crate::op::mean_dim::MeanDim;
        ($t, std::marker::PhantomData::<$i>).mean_dim()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::mean_dim![$crate::mean_dim![$t,$i],$($is),+]
    }};
}

pub trait MeanDim {
    type Out;
    fn mean_dim(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> MeanDim for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Tensor<BTensor<B, N>, <(S, Dim, U0, U1) as narrow::Compatible>::Out>;
    fn mean_dim(self) -> Self::Out {
        Tensor(
            self.0.into_inner().mean_dim(<Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/mod.rs
================================================
pub mod argmax;
pub mod cat;
pub mod expand;
pub mod flatten;
pub mod gather;
pub mod log_softmax;
pub mod matmul;
pub mod mean_dim;
pub mod narrow;
pub mod reshape;
pub mod softmax;
pub mod sort_descending_with_indices;
pub mod squeeze;
pub mod transpose;
pub mod tril_mask;
pub mod unsqueeze;
pub mod var_mean;



================================================
FILE: glowstick-burn/src/op/narrow.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{BasicOps, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{
    num::Unsigned,
    op::{narrow, narrow_dyn, narrow_dyn_start},
    Shape,
};

use crate::Tensor;

/// Narrows a tensor at the specified dimension from start index to length.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{narrow, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let narrowed = narrow!(a.clone(), U0: [U1, U1]);
///
/// assert_eq!(narrowed.dims(), [1, 3, 4]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic start and length, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{narrow, Tensor};
/// use glowstick::{Shape3, num::{U0, U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     N: SequenceLength
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let [start, len] = [1, 2];
/// let narrowed = narrow!(a.clone(), U1: [{ start }, { len }] => N);
///
/// assert_eq!(narrowed.dims(), [2, 2, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! narrow {
    ($t:expr,$d:ty:[$s:ty,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>
        ).narrow()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn()
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        let narrowed = (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>,
        )
            .narrow();
        $crate::narrow!(narrowed,$($ds)+)
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty,$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow::NarrowDyn;
        let narrowed = (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn();
        $crate::narrow!(narrowed,$($ds)+)
    }};
}

pub trait Narrow {
    type Out;
    fn narrow(self) -> Self::Out;
}
impl<B, Dtype, S, Dim, Start, Len, const N: usize> Narrow
    for (
        Tensor<BTensor<B, N, Dtype>, S>,
        PhantomData<Dim>,
        PhantomData<Start>,
        PhantomData<Len>,
    )
where
    B: Backend,
    Dtype: TensorKind<B> + BasicOps<B>,
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    Start: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type Out = Tensor<BTensor<B, N, Dtype>, <(S, Dim, Start, Len) as narrow::Compatible>::Out>;
    fn narrow(self) -> Self::Out {
        Tensor(
            self.0.into_inner().narrow(
                <Dim as glowstick::num::Unsigned>::USIZE,
                <Start as glowstick::num::Unsigned>::USIZE,
                <Len as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}

pub trait NarrowDynStart<const N: usize> {
    type Out;
    fn narrow_dyn_start(self) -> Self::Out;
}
impl<B, Dtype, S, Dim, Len, const N: usize> NarrowDynStart<N>
    for (
        Tensor<BTensor<B, N, Dtype>, S>,
        PhantomData<Dim>,
        usize,
        PhantomData<Len>,
    )
where
    S: Shape,
    B: Backend,
    Dtype: TensorKind<B> + BasicOps<B>,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type Out = Tensor<BTensor<B, N, Dtype>, <(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>;
    fn narrow_dyn_start(self) -> Self::Out {
        Tensor(
            self.0.into_inner().narrow(
                <Dim as glowstick::num::Unsigned>::USIZE,
                self.2,
                <Len as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}

pub trait NarrowDyn {
    type Out;
    fn narrow_dyn(self) -> Self::Out;
}
impl<B, Dtype, S, Dim, DynDim, const N: usize> NarrowDyn
    for (
        Tensor<BTensor<B, N, Dtype>, S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    B: Backend,
    S: Shape,
    Dtype: TensorKind<B> + BasicOps<B>,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Tensor<BTensor<B, N, Dtype>, <(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>;
    fn narrow_dyn(self) -> Self::Out {
        Tensor(
            self.0
                .into_inner()
                .narrow(<Dim as glowstick::num::Unsigned>::USIZE, self.3, self.4),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/reshape.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{BasicOps, ReshapeArgs, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::ShapeFragment;
use glowstick::{op::reshape, Shape, TensorShape};

use crate::Tensor;

/// Reshapes a tensor to the specified dimensions.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{reshape, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U2, U3>>::ones(&device);
/// let reshaped = reshape!(a.clone(), [U1, U6]);
///
/// assert_eq!(reshaped.dims(), [1, 6]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic dimensions, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{reshape, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     A: Rows,
///     B: Cols
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 2>, Shape2<U1, U4>>::ones(&device);
/// let [rows, cols] = [2, 2];
/// let reshaped = reshape!(a.clone(), [{ rows } => A, { cols } => B]);
///
/// assert_eq!(reshaped.dims(), [2, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! reshape {
    ($t:expr,[$($ds:tt)+]) => {{
        type TS = glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        glowstick::op::reshape::check::<_, _, TS>(&$t);
        use $crate::op::reshape::Reshape;
        (
            $t,
            std::marker::PhantomData::<TS>,
        )
            .reshape($crate::reshape_val!($($ds)+).into_array())
    }};
}
#[macro_export]
macro_rules! reshape_tys {
    ($e:expr => $d:ty) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, glowstick::Empty)>
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, $crate::reshape_tys!($($ds)+))>
    };
    ($d:ty) => {
        glowstick::Shp<($d, glowstick::Empty)>
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::Shp<($d, $crate::reshape_tys!($($ds)+))>
    };
}
#[macro_export]
macro_rules! reshape_val {
    ($e:expr => $d:ty) => {
        glowstick::ValueList(($e, glowstick::ValueList(())))
    };
    ($d:ty) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::I32,glowstick::ValueList(())))
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::ValueList(($e,$crate::reshape_val!($($ds)+)))
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::I32,$crate::reshape_val!($($ds)+)))
    };
}

pub trait Reshape<Args, const M: usize> {
    type Out;
    fn reshape(self, args: Args) -> Self::Out;
}
impl<B, D, S1, S2, Args, const N: usize, const M: usize> Reshape<Args, M>
    for (Tensor<BTensor<B, N, D>, S1>, PhantomData<TensorShape<S2>>)
where
    Args: ReshapeArgs<M>,
    B: Backend,
    D: TensorKind<B> + BasicOps<B>,
    S1: Shape,
    TensorShape<S2>: Shape,
    S2: ShapeFragment,
    (S1, TensorShape<S2>): reshape::Compatible,
{
    type Out = Tensor<BTensor<B, M, D>, TensorShape<S2>>;
    fn reshape(self, args: Args) -> Self::Out {
        Tensor(self.0.into_inner().reshape(args), PhantomData)
    }
}



================================================
FILE: glowstick-burn/src/op/softmax.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor as BTensor;
use glowstick::cmp::Greater;
use glowstick::{num::Unsigned, Shape};

use crate::Tensor;

/// Applies the softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let softmaxed = softmax!(a, U1);
///
/// assert_eq!(softmaxed.dims(), [2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! softmax {
    [$t:expr,$i:ty] => {{
        use $crate::op::softmax::Softmax;
        ($t, std::marker::PhantomData::<$i>).softmax()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::softmax![$crate::softmax![$t,$i],$($is),+]
    }};
}

pub trait Softmax {
    type Out;
    fn softmax(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> Softmax for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Tensor<BTensor<B, N>, S>;
    fn softmax(self) -> Self::Out {
        Tensor(
            softmax(self.0.into_inner(), <Dim as Unsigned>::USIZE),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/sort_descending_with_indices.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{
    BasicOps, Int, Numeric,
    Tensor as BTensor,
};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::cmp::Greater;
use glowstick::{
    num::Unsigned, Shape,
};

use crate::Tensor;

/// Applies the sort-descending operation with indices to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{sort_descending_with_indices, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let (sorted, indices) = sort_descending_with_indices!(a, U1);
///
/// assert_eq!(sorted.dims(), [2, 3, 4]);
/// assert_eq!(indices.dims(), [2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! sort_descending_with_indices {
    [$t:expr,$i:ty] => {{
        use $crate::op::sort_descending_with_indices::SortDescendingWithIndices;
        ($t, std::marker::PhantomData::<$i>).sort_descending_with_indices()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::sort_descending_with_indices![$crate::sort_descending_with_indices![$t,$i],$($is),+]
    }};
}

pub trait SortDescendingWithIndices {
    type Out;
    fn sort_descending_with_indices(self) -> Self::Out;
}
impl<B, D, S, const N: usize, Dim> SortDescendingWithIndices
    for (Tensor<BTensor<B, N, D>, S>, PhantomData<Dim>)
where
    B: Backend,
    D: TensorKind<B> + BasicOps<B> + Numeric<B>,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = (Tensor<BTensor<B, N, D>, S>, Tensor<BTensor<B, N, Int>, S>);
    fn sort_descending_with_indices(self) -> Self::Out {
        let (t, i) = self
            .0
            .into_inner()
            .sort_descending_with_indices(<Dim as Unsigned>::USIZE);
        (Tensor(t, PhantomData), Tensor(i, PhantomData))
    }
}



================================================
FILE: glowstick-burn/src/op/squeeze.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{BasicOps, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{num::Unsigned, op::squeeze, Shape};

use crate::Tensor;

/// Squeezes the specified dimensions from a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{squeeze, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 4>, Shape4<U1, U2, U3, U1>>::ones(&device);
/// let squeezed = squeeze![a, U0, U3]; // Squeezes dimensions 0 and 3
///
/// assert_eq!(squeezed.dims(), [2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        glowstick::op::squeeze::check::<_, _, $i>(&$t);
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::squeeze_next![$crate::squeeze![$t,$i],$($is),+]
    }};
}
#[macro_export]
macro_rules! squeeze_next {
    [$t:expr,$i:ty] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<<$i as std::ops::Sub<glowstick::num::U1>>::Output>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::squeeze_next![$crate::squeeze_next![$t,$i],$($is),+]
    }};
}

pub trait Squeeze<const M: usize> {
    type Out;
    fn squeeze(self) -> Self::Out;
}
macro_rules! squeeze_impl {
    ($in:literal => $out:literal) => {
        impl<B, D, S, Dim> Squeeze<$out> for (Tensor<BTensor<B, $in, D>, S>, PhantomData<Dim>)
        where
            B: Backend,
            D: TensorKind<B> + BasicOps<B>,
            S: Shape,
            Dim: Unsigned,
            (S, Dim): squeeze::Compatible,
        {
            type Out = Tensor<BTensor<B, $out, D>, <(S, Dim) as squeeze::Compatible>::Out>;
            fn squeeze(self) -> Self::Out {
                Tensor::<BTensor<B, $out, D>, <(S, Dim) as squeeze::Compatible>::Out>(
                    self.0.into_inner().squeeze(<Dim as Unsigned>::USIZE),
                    PhantomData,
                )
            }
        }
    };
}
squeeze_impl!(8 => 7);
squeeze_impl!(7 => 6);
squeeze_impl!(6 => 5);
squeeze_impl!(5 => 4);
squeeze_impl!(4 => 3);
squeeze_impl!(3 => 2);
squeeze_impl!(2 => 1);



================================================
FILE: glowstick-burn/src/op/transpose.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::Tensor as BTensor;

use glowstick::{num::Unsigned, op::transpose, Shape};

use crate::Tensor;

/// Swaps the dimensions of a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{transpose, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let transposed = transpose!(a, U1, U2);
///
/// assert_eq!(transposed.dims(), [2, 4, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose().and_then(|t| $crate::transpose!(&t, $($d1s:$d2s),+))
    }};
}

pub trait Transpose {
    type Out;
    fn transpose(self) -> Self::Out;
}
impl<B, S, Dim1, Dim2, const N: usize> Transpose
    for (
        Tensor<BTensor<B, N>, S>,
        PhantomData<Dim1>,
        PhantomData<Dim2>,
    )
where
    B: Backend,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Tensor<BTensor<B, N>, <(S, Dim1, Dim2) as transpose::Compatible>::Out>;
    fn transpose(self) -> Self::Out {
        Tensor(
            self.0.into_inner().swap_dims(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            ),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/tril_mask.rs
================================================
use std::marker::PhantomData;

use burn::prelude::Backend;
use burn::tensor::{Bool, Tensor as BTensor};

use glowstick::Shape;

use crate::Tensor;

/// Creates a lower triangular mask of the specified shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{tril_mask, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = NdArrayDevice::Cpu;
/// let mask = tril_mask!(0, &device, Backend, [{ 2 } => B, { 3 } => N]);
///
/// assert_eq!(
///     mask.inner().to_data().to_vec::<bool>().unwrap(),
///     &[
///         false, true, true,
///         false, false, true
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! tril_mask {
    ($o:expr,$d:expr,$b:ty,[$($ds:tt)+]) => {{
        use $crate::op::tril_mask::TrilMask;
        (
            std::marker::PhantomData::<$b>,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
            $o,
            $d,
        )
            .tril_mask($crate::reshape_val!($($ds)+).into_array())
    }};
}

pub trait TrilMask<const M: usize> {
    type Out;
    fn tril_mask(self, shape: [usize; M]) -> Self::Out;
}
impl<B, S, const M: usize> TrilMask<M>
    for (PhantomData<B>, PhantomData<S>, i64, &<B as Backend>::Device)
where
    B: Backend,
    S: Shape,
{
    type Out = Tensor<BTensor<B, M, Bool>, S>;
    fn tril_mask(self, shape: [usize; M]) -> Self::Out {
        Tensor(
            BTensor::<B, M, Bool>::tril_mask(shape, self.2, self.3),
            PhantomData,
        )
    }
}



================================================
FILE: glowstick-burn/src/op/unsqueeze.rs
================================================
use std::marker::PhantomData;

use burn::tensor::{BasicOps, Tensor as BTensor};
use burn::{prelude::Backend, tensor::TensorKind};

use glowstick::{num::Unsigned, op::unsqueeze, Shape};

use crate::Tensor;

/// Unsqueezes a tensor at the specified dimension(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{unsqueeze, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U1, U4>>::ones(&device);
/// let unsqueezed = unsqueeze![a.clone(), U0, U4];
///
/// assert_eq!(unsqueezed.dims(), [1, 2, 1, 4, 1]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::unsqueeze![$crate::unsqueeze![$t,$i],$($is),+]
    }};
}

pub trait Unsqueeze<const M: usize> {
    type Out;
    fn unsqueeze(self) -> Self::Out;
}
macro_rules! unsqueeze_impl {
    ($in:literal => $out:literal) => {
        impl<B, S, D, Dim> Unsqueeze<$out> for (Tensor<BTensor<B, $in, D>, S>, PhantomData<Dim>)
        where
            B: Backend,
            S: Shape,
            D: TensorKind<B> + BasicOps<B>,
            Dim: Unsigned,
            (S, Dim): unsqueeze::Compatible,
        {
            type Out = Tensor<BTensor<B, $out, D>, <(S, Dim) as unsqueeze::Compatible>::Out>;
            fn unsqueeze(self) -> Self::Out {
                Tensor::<BTensor<B, $out, D>, <(S, Dim) as unsqueeze::Compatible>::Out>(
                    self.0.into_inner().unsqueeze_dim(<Dim as Unsigned>::USIZE),
                    PhantomData,
                )
            }
        }
    };
}
unsqueeze_impl!(7 => 8);
unsqueeze_impl!(6 => 7);
unsqueeze_impl!(5 => 6);
unsqueeze_impl!(4 => 5);
unsqueeze_impl!(3 => 4);
unsqueeze_impl!(2 => 3);
unsqueeze_impl!(1 => 2);



================================================
FILE: glowstick-burn/src/op/var_mean.rs
================================================
use std::marker::PhantomData;

use burn::tensor::Tensor as BTensor;
use burn::prelude::Backend;

use glowstick::cmp::Greater;
use glowstick::{
    num::Unsigned, Shape,
};

use crate::Tensor;

/// Computes the variance and mean of a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_burn::Error> {
/// # use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// # type Backend = NdArray;
/// use burn::tensor::{Device, Tensor as BurnTensor};
/// use glowstick_burn::{var_mean, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = NdArrayDevice::Cpu;
/// let a = Tensor::<BurnTensor<Backend, 3>, Shape3<U2, U3, U4>>::ones(&device);
/// let (variance, mean) = var_mean!(a, U1);
///
/// assert_eq!(variance.dims(), [2, 1, 4]);
/// assert_eq!(mean.dims(), [2, 1, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! var_mean {
    [$t:expr,$i:ty] => {{
        use $crate::op::var_mean::VarMean;
        ($t, std::marker::PhantomData::<$i>).var_mean()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::var_mean![$crate::var_mean![$t,$i],$($is),+]
    }};
}

pub trait VarMean {
    type Out;
    fn var_mean(self) -> Self::Out;
}
impl<B, S, const N: usize, Dim> VarMean for (Tensor<BTensor<B, N>, S>, PhantomData<Dim>)
where
    B: Backend,
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = (Tensor<BTensor<B, N>, S>, Tensor<BTensor<B, N>, S>);
    fn var_mean(self) -> Self::Out {
        let (var, mean) = self.0.into_inner().var_mean(<Dim as Unsigned>::USIZE);
        (Tensor(var, PhantomData), Tensor(mean, PhantomData))
    }
}



================================================
FILE: glowstick-candle/Cargo.toml
================================================
[package]
name = "glowstick-candle"
description = "Integration of glowstick with the candle tensor"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
categories.workspace = true
keywords.workspace = true

[dependencies]
candle = { version = "0.9", package = "candle-core" }
candle-nn = { version = "0.9" }
glowstick.workspace = true
thiserror.workspace = true




================================================
FILE: glowstick-candle/src/lib.rs
================================================
pub mod op;
pub mod tensor;

pub use tensor::{Error, Tensor};

#[doc = include_str!("../../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;



================================================
FILE: glowstick-candle/src/tensor.rs
================================================
use std::marker::PhantomData;

use candle::{DType, Device};

use glowstick::Shape;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Rank mismatch: runtime ({runtime}) vs type-level ({type_level})")]
    RankMismatch { runtime: usize, type_level: usize },

    #[error("Dimension mismatch: expected {type_level} for dim {dim} but received {runtime}")]
    DimensionMismatch {
        dim: usize,
        runtime: usize,
        type_level: usize,
    },

    #[error("{0}")]
    Candle(#[from] candle::Error),
}

#[allow(unused)]
#[derive(Debug)]
pub struct Tensor<S: Shape>(pub(crate) candle::Tensor, pub(crate) PhantomData<S>);
impl<S: Shape> AsRef<candle::Tensor> for Tensor<S> {
    fn as_ref(&self) -> &candle::Tensor {
        self.inner()
    }
}

impl<S: Shape> Clone for Tensor<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<S> glowstick::Tensor for Tensor<S>
where
    S: Shape,
{
    type Shape = S;
}
impl<S> glowstick::Tensor for &Tensor<S>
where
    S: Shape,
{
    type Shape = S;
}

impl<S> TryFrom<candle::Tensor> for Tensor<S>
where
    S: Shape,
{
    type Error = Error;
    fn try_from(x: candle::Tensor) -> Result<Self, Self::Error> {
        if S::RANK != x.rank() {
            return Err(Error::RankMismatch {
                runtime: x.rank(),
                type_level: S::RANK,
            });
        }

        for (dim, (a, b)) in x.dims().iter().copied().zip(S::iter()).enumerate() {
            if a != b {
                return Err(Error::DimensionMismatch {
                    dim,
                    runtime: a,
                    type_level: b,
                });
            }
        }

        Ok(Self(x, PhantomData))
    }
}

impl<S> std::ops::Add<Tensor<S>> for &Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 + rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Add<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((self.0 + rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Add<&Tensor<S>> for &Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 + &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Add<&Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn add(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((self.0 + &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Sub<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn sub(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 - rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Mul<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Mul<&Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Mul<&Tensor<S>> for &Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: &Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * &rhs.0)?, PhantomData))
    }
}

impl<S> std::ops::Mul<f64> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn mul(self, rhs: f64) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 * rhs)?, PhantomData))
    }
}

impl<S> std::ops::Div<Tensor<S>> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn div(self, rhs: Tensor<S>) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 / &rhs.0)?, PhantomData))
    }
}
impl<S> std::ops::Div<f64> for Tensor<S>
where
    S: Shape,
{
    type Output = Result<Tensor<S>, Error>;
    fn div(self, rhs: f64) -> Result<Tensor<S>, Error> {
        Ok(Tensor((&self.0 / rhs)?, PhantomData))
    }
}

impl<S> Tensor<S>
where
    S: Shape,
{
    pub fn inner(&self) -> &candle::Tensor {
        &self.0
    }

    pub fn shape() -> impl Into<candle::Shape> {
        <S as Shape>::iter().collect::<Vec<_>>()
    }

    pub fn dims(&self) -> &[usize] {
        self.inner().dims()
    }

    pub fn from_vec<D: candle::WithDType>(v: Vec<D>, device: &Device) -> Result<Self, Error> {
        candle::Tensor::from_vec(v, Self::shape(), device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    pub fn zeros(dtype: DType, device: &Device) -> Result<Self, Error> {
        candle::Tensor::zeros(Self::shape(), dtype, device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    pub fn ones(dtype: DType, device: &Device) -> Result<Self, Error> {
        candle::Tensor::ones(Self::shape(), dtype, device)
            .map(|t| Self(t, PhantomData))
            .map_err(Into::into)
    }

    pub fn zeros_like(&self) -> Result<Self, Error> {
        Ok(Self(self.0.zeros_like()?, PhantomData))
    }

    /// Return the candle tensor, discarding type information
    pub fn into_inner(self) -> candle::Tensor {
        self.0
    }

    pub fn to_dtype(&self, dtype: candle::DType) -> Result<Self, Error> {
        Ok(Self(self.0.to_dtype(dtype)?, PhantomData))
    }

    pub fn dtype(&self) -> candle::DType {
        self.0.dtype()
    }

    pub fn contiguous(&self) -> Result<Self, Error> {
        Ok(Self(self.0.contiguous()?, PhantomData))
    }

    pub fn exp(&self) -> Result<Self, Error> {
        Ok(Self(self.0.exp()?, PhantomData))
    }

    pub fn clamp(&self, a: f32, b: f32) -> Result<Self, Error> {
        Ok(Self(self.0.clamp(a, b)?, PhantomData))
    }

    pub fn neg(&self) -> Result<Self, Error> {
        Ok(Self(self.0.neg()?, PhantomData))
    }

    pub fn to_device(&self, device: &Device) -> Result<Self, Error> {
        Ok(Self(self.0.to_device(device)?, PhantomData))
    }

    pub fn log(&self) -> Result<Self, Error> {
        Ok(Self(self.0.log()?, PhantomData))
    }

    pub fn minimum(&self, other: &Self) -> Result<Self, Error> {
        Ok(Self(self.0.minimum(other.inner())?, PhantomData))
    }

    pub fn maximum(&self, other: &Self) -> Result<Self, Error> {
        Ok(Self(self.0.maximum(other.inner())?, PhantomData))
    }

    pub fn detach(self) -> Self {
        Self(self.0.detach(), PhantomData)
    }

    pub fn abs(&self) -> Result<Self, Error> {
        Ok(Self(self.0.abs()?, PhantomData))
    }
}



================================================
FILE: glowstick-candle/src/op/broadcast_add.rs
================================================
use std::borrow::Borrow;

use glowstick::{op::broadcast, Shape};

use crate::{Error, Tensor};

/// Performs addition of the lefthand tensor and righthand tensor(s). The righthand
/// tensor(s) must be compatible for broadcast to the shape of the lefthand tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{broadcast_add, Tensor};
/// use glowstick::{Shape1, Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape2<U2, U2>>::ones(DType::F32, &device)?;
/// let c = broadcast_add!(a, b)?;
///
/// assert_eq!(
///     c.inner().to_vec2::<f32>()?,
///     vec![
///         vec![2., 2.],
///         vec![2., 2.]
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! broadcast_add {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::broadcast_add::BroadcastAdd;
        ($t1, $t2).broadcast_add()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::op::broadcast_add::BroadcastAdd;
        ($t1, $t2)
            .broadcast_add()
            .and_then(|t| $crate::broadcast_add!(&t, $t2s))
    }};
}

pub trait BroadcastAdd {
    type Out;
    fn broadcast_add(&self) -> Self::Out;
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0
            .inner()
            .broadcast_add(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0
            .inner()
            .broadcast_add(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/cat.rs
================================================
use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::cat_dyn, Shape};

use crate::{Error, Tensor};

/// Concatenates the given tensors along a specified dimension.
/// A dynamic dimension must be provided for the return type.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{cat, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let concatenated = cat!(vec![a, b].as_slice(), U0 => B)?;
///
/// assert_eq!(concatenated.dims(), &[2, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! cat {
    ($ts:expr,$i:ty => $d:ty) => {{
        use $crate::op::cat::Cat;
        (
            $ts,
            std::marker::PhantomData::<$i>,
            std::marker::PhantomData::<$d>,
        )
            .cat()
    }};
}

pub trait Cat {
    type Out;
    fn cat(self) -> Self::Out;
}
impl<S, I, D> Cat for (&[Tensor<S>], PhantomData<I>, PhantomData<glowstick::Dyn<D>>)
where
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn cat(self) -> Self::Out {
        candle::Tensor::cat(self.0, <I as Unsigned>::USIZE)?.try_into()
    }
}
impl<S, I, D> Cat
    for (
        &[&Tensor<S>],
        PhantomData<I>,
        PhantomData<glowstick::Dyn<D>>,
    )
where
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn cat(self) -> Self::Out {
        candle::Tensor::cat(self.0, <I as Unsigned>::USIZE)?.try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/conv.rs
================================================
use std::marker::PhantomData;

use crate::{Error, Tensor};
use glowstick::op::convolution::IsCompatible;
use glowstick::{
    num::{Unsigned, U0},
    op::convolution,
    Indexed, Shape, ShapeDiagnostic, ShapeFragment,
};

/// Applies a 2D convolution over the input tensor with the provided kernel, padding,
/// dilation, stride and groups.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{conv2d, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = Device::Cpu;
/// let input = Tensor::<Shape4<U2, U2, U5, U5>>::ones(DType::F32, &device)?;
/// let kernel = Tensor::<Shape4<U4, U2, U3, U3>>::ones(DType::F32, &device)?;
/// let convolved = conv2d!(input, kernel, U0, U1, U1, 1)?;
///
/// assert_eq!(convolved.dims(), &[2, 4, 3, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! conv2d {
    ($t:expr,$kernel:expr,$padding:ty,$dilation:ty,$stride:ty,$groups:expr) => {{
        use std::marker::PhantomData;
        use $crate::op::conv::Conv2d;
        type Pad = glowstick::list![$padding, $padding];
        type Dilation = glowstick::list![$dilation, $dilation];
        type Stride = glowstick::list![$stride, $stride];
        (
            $t,
            $kernel,
            PhantomData::<Pad>,
            PhantomData::<Pad>,
            PhantomData::<Dilation>,
            PhantomData::<Stride>,
            $groups,
        )
            .conv2d()
    }};
}

pub trait Conv2d {
    type Out;
    fn conv2d(self) -> Self::Out;
}

use convolution::Kernel;
use glowstick::num::U1;
use glowstick::num::{Sub, ZipSubOneMul};
use glowstick::{Container, Empty, List, Map, Mappend, TakeFragment};

impl<T, K, P1, P2, S, D> Conv2d
    for (
        Tensor<T>,
        Tensor<K>,
        PhantomData<P1>,
        PhantomData<P2>,
        PhantomData<D>,
        PhantomData<S>,
        usize,
    )
where
    (T, K, P1, P2, S, D): convolution::IsCompatible,
    (P1, U0): Indexed,
    (S, U0): Indexed,
    (D, U0): Indexed,
    <(P1, U0) as Indexed>::Out: Unsigned,
    <(S, U0) as Indexed>::Out: Unsigned,
    <(D, U0) as Indexed>::Out: Unsigned,
    T: Shape + ShapeDiagnostic,
    K: Kernel<D> + ShapeDiagnostic,
    (T, K, P1, P2, S, D): IsCompatible,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    <K as Kernel<D>>::DilateZipped: Container,
    (<K as Kernel<D>>::DilateZipped, ZipSubOneMul):
        Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        List<(<K as Kernel<D>>::M, Empty)>,
    ): Mappend,
    (T, K, P1, P2, S, D): convolution::Compatible,
{
    type Out = Result<Tensor<<(T, K, P1, P2, S, D) as convolution::Compatible>::Out>, Error>;

    fn conv2d(self) -> Self::Out {
        let p = <<(P1, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        let s = <<(S, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        let d = <<(D, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        self.0
            .inner()
            .conv2d(self.1.inner(), p, s, d, self.6)?
            .try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/expand.rs
================================================
use std::marker::PhantomData;

use glowstick::{op::broadcast, Shape};

use crate::{Error, Tensor};

/// Broadcasts the lefthand tensor to the shape of the provided righthand tensor
/// or shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let c = expand!(&a, &b)?;
/// let d = expand!(&a, [U1, U4, U3, U2])?;
///
/// assert_eq!(c.dims(), &[1, 4, 3, 2]);
/// assert_eq!(d.dims(), &[1, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
///
/// When broadcasting to a shape, a combination of type-level integers and
/// expressions bound to dynamic dimensions may be provided.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::{U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = Device::Cpu;
/// let [batch_size, seq_len] = [4, 12];
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = expand!(&a, [{ batch_size } => B, { seq_len } => N, U3, U2])?;
///
/// assert_eq!(b.dims(), &[4, 12, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! expand {
    ($t:expr,[$($ds:tt)+]) => {{
        use $crate::op::expand::BroadcastAs;
        (
            $t,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
        )
            .expand(&candle::Shape::from_dims(&$crate::reshape_val!($($ds)+).into_array()))
    }};
    ($t1:expr,$t2:expr) => {{
        use $crate::op::expand::BroadcastAs;
        (
            $t1,
            $t2,
        )
            .expand($t2.inner().shape())
    }}
}

pub trait BroadcastAs {
    type Out;
    fn expand(&self, shape: &candle::Shape) -> Self::Out;
}

impl<S1, S2> BroadcastAs for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, PhantomData<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &candle::Shape) -> Self::Out {
        self.0.inner().expand(shape)?.try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/flatten.rs
================================================
use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::flatten, Shape};

use crate::{Error, Tensor};

/// Flattens the given tensor from the specified start dimension to the end
/// dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{flatten, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let flattened = flatten!(a, [U0, U2])?;
///
/// assert_eq!(flattened.dims(), &[12, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! flatten {
    ($t:expr,[$d1:ty,$d2:ty]) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten()
    }};
    ($t:expr,[$d1:ty,$d2:ty],$([$d1s:ty,$d2s:ty]),+) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten().and_then(|t| $crate::flatten!(&t, $([$d1s,$d2s]),+))
    }};
}

pub trait Flatten {
    type Out;
    fn flatten(&self) -> Self::Out;
}
impl<S, Dim1, Dim2> Flatten for (Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}
impl<S, Dim1, Dim2> Flatten for (&Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/gather.rs
================================================
use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::gather, Shape};

use crate::{Error, Tensor};

/// Gathers the elements from a tensor at the provided indices along a specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{gather, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U1, U1, U4>>::from_vec(vec![1f32, 2., 3., 4.], &device)?;
/// let b = Tensor::<Shape3<U1, U1, U2>>::from_vec(vec![1u32, 2], &device)?;
/// let gathered = gather!(a, b, U2)?;
///
/// assert_eq!(gathered.inner().to_vec3::<f32>()?, vec![vec![vec![2., 3.]]]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! gather {
    ($t1:expr,$t2:expr,$d:ty) => {{
        use $crate::op::gather::Gather;
        (
            $t1,
            std::marker::PhantomData,
            $t2,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
        )
            .gather()
    }};
}

pub trait Gather {
    type Out;
    fn gather(&self) -> Self::Out;
}
impl<T1, S1, T2, S2, Dim> Gather for (T1, PhantomData<S1>, T2, PhantomData<S2>, PhantomData<Dim>)
where
    T1: Borrow<Tensor<S1>>,
    S1: Shape,
    T2: Borrow<Tensor<S2>>,
    S2: Shape,
    Dim: Unsigned,
    (S1, S2, Dim): gather::Compatible,
{
    type Out = Result<Tensor<<(S1, S2, Dim) as gather::Compatible>::Out>, Error>;
    fn gather(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .gather(self.2.borrow().inner(), <Dim as Unsigned>::USIZE)?
            .try_into()
    }
}

#[cfg(test)]
mod test_gather {
    #[test]
    fn gather() {
        use crate::gather;
        use crate::Tensor;
        use glowstick::num::{U1, U2, U4};
        type A = glowstick::shape![U1, U1, U4];
        type B = glowstick::shape![U1, U1, U2];
        let a: Tensor<A> = Tensor::from_vec(vec![1f32, 2., 3., 4.], &candle::Device::Cpu).unwrap();
        let b: Tensor<B> = Tensor::from_vec(vec![1u32, 2], &candle::Device::Cpu).unwrap();
        let gathered = gather!(a, b, U2).unwrap();
        let v = gathered.into_inner().to_vec3::<f32>().unwrap();
        assert_eq!(v, vec![vec![vec![2., 3.]]]);
    }
}



================================================
FILE: glowstick-candle/src/op/log_softmax.rs
================================================
use std::marker::PhantomData;

use glowstick::cmp::Greater;
use glowstick::num::Unsigned;
use glowstick::Shape;

use crate::Tensor;

/// Applies the log softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{log_softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let logsoftmaxed = log_softmax!(a, U1)?;
///
/// assert_eq!(logsoftmaxed.dims(), &[2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! log_softmax {
    ($t:expr,$i:ty) => {{
        use $crate::op::log_softmax::LogSoftmax;
        ($t, std::marker::PhantomData::<$i>).log_softmax()
    }};
    ($t:expr,$i:ty,$($is:ty),+) => {{
        $crate::log_softmax!($crate::log_softmax!($t,$i),$($is),+)
    }};
}

pub trait LogSoftmax {
    type Out;
    fn log_softmax(self) -> Self::Out;
}
impl<S, Dim> LogSoftmax for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn log_softmax(self) -> Self::Out {
        Ok(Tensor(
            candle_nn::ops::log_softmax(self.0.inner(), <Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}

#[cfg(test)]
mod test_logsoft {
    #[test]
    fn logsoft() {
        use crate::log_softmax;
        use crate::Tensor;
        use glowstick::num::{U0, U4};
        type TestShape = glowstick::shape![U4];
        let ct = candle::Tensor::from_vec(vec![0., 1., 2., 3.], 4, &candle::Device::Cpu).unwrap();
        let gt: Tensor<TestShape> = ct.clone().try_into().unwrap();
        let c_softmaxed: Vec<f64> = candle_nn::ops::log_softmax(&ct, 0)
            .unwrap()
            .to_vec1()
            .unwrap();
        let g_softmaxed: Vec<f64> = log_softmax!(gt, U0)
            .unwrap()
            .into_inner()
            .to_vec1()
            .unwrap();
        assert_eq!(c_softmaxed, g_softmaxed);
    }
}



================================================
FILE: glowstick-candle/src/op/matmul.rs
================================================
use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{op::matmul, Shape, TensorShape};

use crate::{Error, Tensor};

/// Performs matrix multiplication of the lefthand tensor and righthand tensor(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{matmul, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U1>>::from_vec(vec![4f32, 5.], &device)?;
/// let b = Tensor::<Shape2<U1, U2>>::from_vec(vec![5f32, 4.], &device)?;
/// let c = matmul!(a, b)?;
///
/// assert_eq!(
///     c.inner().to_vec2::<f32>()?,
///     vec![
///         vec![20., 16.],
///         vec![25., 20.]
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2, std::marker::PhantomData).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2, std::marker::PhantomData)
            .matmul()
            .and_then(|t| $crate::matmul!(&t, $t2s))
    }};
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<S1, U, S2> Matmul for (Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0
            .into_inner()
            .matmul(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, U, S2> Matmul for (&Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0.inner().matmul(self.1.borrow().inner())?.try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/mod.rs
================================================
pub mod broadcast_add;
pub mod cat;
pub mod conv;
pub mod expand;
pub mod flatten;
pub mod gather;
pub mod log_softmax;
pub mod matmul;
pub mod narrow;
pub mod narrow_dyn;
pub mod narrow_dyn_start;
pub mod reshape;
pub mod softmax;
pub mod squeeze;
pub mod transpose;
pub mod unsqueeze;



================================================
FILE: glowstick-candle/src/op/narrow.rs
================================================
use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::narrow, Shape};

use crate::{Error, Tensor};

/// Narrows a tensor at the specified dimension from start index to length.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{narrow, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let narrowed = narrow!(a, U0: [U1, U1])?;
///
/// assert_eq!(narrowed.dims(), &[1, 3, 4]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic start and length, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{narrow, Tensor};
/// use glowstick::{Shape3, num::{U0, U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     N: SequenceLength
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let [start, len] = [1, 2];
/// let narrowed = narrow!(a, U1: [{ start }, { len }] => N)?;
///
/// assert_eq!(narrowed.dims(), &[2, 2, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! narrow {
    ($t:expr,$d:ty:[$s:ty,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>
        ).narrow()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty]) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow_dyn_start::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow_dyn::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn()
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>,
        )
            .narrow().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow_dyn_start::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty,$($ds:tt)+) => {{
        glowstick::op::narrow::check::<_, _, $d, glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow_dyn::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
}

#[allow(unused)]
pub trait Narrow {
    type Out;
    fn narrow(&self) -> Self::Out;
}
impl<T, S, Dim, Start, Len> Narrow
    for (
        T,
        PhantomData<S>,
        PhantomData<Dim>,
        PhantomData<Start>,
        PhantomData<Len>,
    )
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Start: Unsigned,
    Len: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Start, Len) as narrow::Compatible>::Out>, Error>;
    fn narrow(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(
                <Dim as Unsigned>::USIZE,
                <Start as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE,
            )?
            .try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/narrow_dyn.rs
================================================
use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::narrow_dyn, Shape};

use crate::{Error, Tensor};

#[allow(unused)]
pub trait NarrowDyn {
    type Out;
    fn narrow_dyn(&self) -> Self::Out;
}
impl<S, Dim, DynDim> NarrowDyn
    for (
        Tensor<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, self.4)?
            .try_into()
    }
}
impl<S, Dim, DynDim> NarrowDyn
    for (
        &Tensor<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, self.4)?
            .try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/narrow_dyn_start.rs
================================================
use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{Shape, num::Unsigned, op::narrow_dyn_start};

use crate::{Error, Tensor};

pub trait NarrowDynStart {
    type Out;
    fn narrow_dyn_start(&self) -> Self::Out;
}
impl<T, S, Dim, Len> NarrowDynStart
    for (T, PhantomData<S>, PhantomData<Dim>, usize, PhantomData<Len>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>, Error>;
    fn narrow_dyn_start(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, <Len as Unsigned>::USIZE)?
            .try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/reshape.rs
================================================
use std::{borrow::Borrow, marker::PhantomData};

use candle::shape::ShapeWithOneHole;
use glowstick::{op::reshape, Shape};

use crate::{Error, Tensor};

/// Reshapes a tensor to the specified dimensions.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{reshape, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let reshaped = reshape!(a, [U1, U6])?;
///
/// assert_eq!(reshaped.dims(), &[1, 6]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic dimensions, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{reshape, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     A: Rows,
///     B: Cols
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U4>>::ones(DType::F32, &device)?;
/// let [rows, cols] = [2, 2];
/// let reshaped = reshape!(a, [{ rows } => A, { cols } => B])?;
///
/// assert_eq!(reshaped.dims(), &[2, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! reshape {
    ($t:expr,[$($ds:tt)+]) => {{
        type TS = glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        glowstick::op::reshape::check::<_, _, TS>(&$t);
        use $crate::op::reshape::Reshape;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<TS>,
        )
            .reshape($crate::reshape_val!($($ds)+).tuplify())
    }};
}

pub trait Reshape {
    type Out;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out;
}

impl<T, S1, S2> Reshape for (T, PhantomData<S1>, PhantomData<S2>)
where
    T: Borrow<Tensor<S1>>,
    S1: Shape,
    S2: Shape,
    (S1, S2): reshape::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as reshape::Compatible>::Out>, Error>;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out {
        self.0.borrow().inner().reshape(args)?.try_into()
    }
}

#[macro_export]
macro_rules! reshape_tys {
    ($e:expr => $d:ty) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, glowstick::Empty)>
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::Shp<(<$d as glowstick::dynamic::Dim>::Id, $crate::reshape_tys!($($ds)+))>
    };
    ($d:ty) => {
        glowstick::Shp<($d, glowstick::Empty)>
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::Shp<($d, $crate::reshape_tys!($($ds)+))>
    };
}
#[macro_export]
macro_rules! reshape_val {
    ($e:expr => $d:ty) => {
        glowstick::ValueList(($e, glowstick::ValueList(())))
    };
    ($d:ty) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::USIZE,glowstick::ValueList(())))
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        glowstick::ValueList(($e,$crate::reshape_val!($($ds)+)))
    };
    ($d:ty,$($ds:tt)+) => {
        glowstick::ValueList((<$d as glowstick::num::Unsigned>::USIZE,$crate::reshape_val!($($ds)+)))
    };
}



================================================
FILE: glowstick-candle/src/op/softmax.rs
================================================
use std::marker::PhantomData;

use glowstick::cmp::Greater;
use glowstick::num::Unsigned;
use glowstick::Shape;

use crate::Tensor;

/// Applies the softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let softmaxed = softmax!(a, U1)?;
///
/// assert_eq!(softmaxed.dims(), &[2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! softmax {
    ($t:expr,$i:ty) => {{
        use $crate::op::softmax::Softmax;
        ($t, std::marker::PhantomData::<$i>).softmax()
    }};
    ($t:expr,$i:ty,$($is:ty),+) => {{
        $crate::softmax!($crate::softmax!($t,$i),$($is),+)
    }};
}

pub trait Softmax {
    type Out;
    fn softmax(self) -> Self::Out;
}
impl<S, Dim> Softmax for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn softmax(self) -> Self::Out {
        Ok(Tensor(
            candle_nn::ops::softmax(self.0.inner(), <Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}



================================================
FILE: glowstick-candle/src/op/squeeze.rs
================================================
use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::squeeze, Shape};

use crate::{Error, Tensor};

/// Squeezes the specified dimensions from a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{squeeze, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U2, U3, U1>>::ones(DType::F32, &device)?;
/// let squeezed = squeeze![a, U0, U3]?; // Squeezes dimensions 0 and 3
///
/// assert_eq!(squeezed.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        glowstick::op::squeeze::check::<_, _, $i>(&$t);
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
            .and_then(|t| $crate::squeeze_next![t, $($is),+])
    }};
}
#[macro_export]
macro_rules! squeeze_next {
    [$t:expr,$i:ty] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<<$i as std::ops::Sub<glowstick::num::U1>>::Output>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
            .and_then(|t| $crate::squeeze_next![t, $($is),+])
    }};
}

pub trait Squeeze {
    type Out;
    fn squeeze(&self) -> Self::Out;
}
impl<S, Dim> Squeeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}
impl<S, Dim> Squeeze for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/transpose.rs
================================================
use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::transpose, Shape};

use crate::{Error, Tensor};

/// Swaps the dimensions of a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{transpose, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let transposed = transpose!(a, U1, U2)?;
///
/// assert_eq!(transposed.dims(), &[2, 4, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose().and_then(|t| $crate::transpose!(&t, $($d1s:$d2s),+))
    }};
}

pub trait Transpose {
    type Out;
    fn transpose(&self) -> Self::Out;
}
impl<T, S, Dim1, Dim2> Transpose for (T, PhantomData<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as transpose::Compatible>::Out>, Error>;
    fn transpose(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .transpose(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}



================================================
FILE: glowstick-candle/src/op/unsqueeze.rs
================================================
use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::unsqueeze, Shape};

use crate::{Error, Tensor};

/// Unsqueezes a tensor at the specified dimension(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), glowstick_candle::Error> {
/// use candle::{Device, DType};
/// use glowstick_candle::{unsqueeze, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U1, U4>>::ones(DType::F32, &device)?;
/// let unsqueezed = unsqueeze![a, U0, U4]?;
///
/// assert_eq!(unsqueezed.dims(), &[1, 2, 1, 4, 1]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
            .and_then(|t| $crate::unsqueeze![t, $($is),+])
    }};
}

#[allow(unused)]
pub trait Unsqueeze {
    type Out;
    fn unsqueeze(&self) -> Self::Out;
}
impl<S, Dim> Unsqueeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}
impl<S, Dim> Unsqueeze for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}



================================================
FILE: src/cmp.rs
================================================
use typosaurus::num::{self, Bit, UInt, UTerm, Unsigned};

use crate::Dyn;
use crate::dynamic::{DynMax, DynMin, IsDynEqual, IsDynGreater, IsDynLess};

pub use typosaurus::bool::{And, Bool, False, Or, True};

pub trait Max {
    type Out;
}
impl<L1, L2> Max for (Dyn<L1>, Dyn<L2>)
where
    L1: DynMax<L2>,
{
    type Out = Dyn<<L1 as DynMax<L2>>::Out>;
}
impl<L> Max for (UTerm, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<L> Max for (Dyn<L>, UTerm) {
    type Out = Dyn<L>;
}
impl<U, B, L> Max for (UInt<U, B>, Dyn<L>)
where
    U: Unsigned,
    B: Bit,
    L: DynMax<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMax<UInt<U, B>>>::Out>;
}
impl<U, B, L> Max for (Dyn<L>, UInt<U, B>)
where
    U: Unsigned,
    B: Bit,
    L: DynMax<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMax<UInt<U, B>>>::Out>;
}
impl<T, U> Max for (T, U)
where
    T: num::Max<U>,
{
    type Out = <T as num::Max<U>>::Output;
}

pub trait Min {
    type Out;
}
impl<L1, L2> Min for (Dyn<L1>, Dyn<L2>)
where
    L1: DynMin<L2>,
{
    type Out = Dyn<<L1 as DynMin<L2>>::Out>;
}
impl<L> Min for (UTerm, Dyn<L>) {
    type Out = UTerm;
}
impl<L> Min for (Dyn<L>, UTerm) {
    type Out = UTerm;
}
impl<U, B, L> Min for (UInt<U, B>, Dyn<L>)
where
    U: Unsigned,
    B: Bit,
    L: DynMin<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMin<UInt<U, B>>>::Out>;
}
impl<U, B, L> Min for (Dyn<L>, UInt<U, B>)
where
    U: Unsigned,
    B: Bit,
    L: DynMin<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMin<UInt<U, B>>>::Out>;
}
impl<T, U> Min for (T, U)
where
    T: num::Min<U>,
{
    type Out = <T as num::Min<U>>::Output;
}

pub trait Equal {
    crate::private!();
}
impl<T, U> Equal for (T, U)
where
    (T, U): IsEqual,
    <(T, U) as IsEqual>::Out: typosaurus::bool::Truthy,
{
    crate::private_impl!();
}

pub trait IsEqual {
    type Out;
}
impl<A, B> IsEqual for (Dyn<A>, Dyn<B>)
where
    A: IsDynEqual<B>,
{
    type Out = <A as IsDynEqual<B>>::Out;
}
impl<L> IsEqual for (UTerm, Dyn<L>) {
    type Out = True;
}
impl<L> IsEqual for (Dyn<L>, UTerm) {
    type Out = True;
}
impl<U, B, L> IsEqual for (UInt<U, B>, Dyn<L>)
where
    L: IsDynEqual<UInt<U, B>>,
{
    type Out = <L as IsDynEqual<UInt<U, B>>>::Out;
}
impl<U, B, L> IsEqual for (Dyn<L>, UInt<U, B>)
where
    L: IsDynEqual<UInt<U, B>>,
{
    type Out = <L as IsDynEqual<UInt<U, B>>>::Out;
}
impl<T, U> IsEqual for (T, U)
where
    T: num::IsEqual<U>,
    <T as num::IsEqual<U>>::Output: Bool,
{
    type Out = <<T as num::IsEqual<U>>::Output as Bool>::Out;
}

pub trait IsLess {
    type Out;
}
impl<L1, L2> IsLess for (Dyn<L1>, Dyn<L2>)
where
    L1: IsDynLess<L2>,
{
    type Out = <L1 as IsDynLess<L2>>::Out;
}
impl<L> IsLess for (UTerm, Dyn<L>) {
    type Out = True;
}
impl<L> IsLess for (Dyn<L>, UTerm) {
    type Out = True;
}
impl<U, B, L> IsLess for (UInt<U, B>, Dyn<L>)
where
    L: IsDynLess<UInt<U, B>>,
{
    type Out = <L as IsDynLess<UInt<U, B>>>::Out;
}
impl<U, B, L> IsLess for (Dyn<L>, UInt<U, B>)
where
    L: IsDynLess<UInt<U, B>>,
{
    type Out = <L as IsDynLess<UInt<U, B>>>::Out;
}
impl<T, U> IsLess for (T, U)
where
    T: num::IsLess<U>,
    <T as num::IsLess<U>>::Output: Bool,
{
    type Out = <<T as num::IsLess<U>>::Output as Bool>::Out;
}

pub trait Greater {
    crate::private!();
}
impl<T, U> Greater for (T, U)
where
    (T, U): IsGreater,
    <(T, U) as IsGreater>::Out: typosaurus::bool::Truthy,
{
    crate::private_impl!();
}

pub trait IsGreater {
    type Out;
}
impl<L1, L2> IsGreater for (Dyn<L1>, Dyn<L2>)
where
    L1: IsDynGreater<L2>,
{
    type Out = <L1 as IsDynGreater<L2>>::Out;
}
impl<L> IsGreater for (UTerm, Dyn<L>) {
    type Out = True;
}
impl<L> IsGreater for (Dyn<L>, UTerm) {
    type Out = True;
}
impl<U, B, L> IsGreater for (UInt<U, B>, Dyn<L>)
where
    L: IsDynGreater<UInt<U, B>>,
{
    type Out = <L as IsDynGreater<UInt<U, B>>>::Out;
}
impl<U, B, L> IsGreater for (Dyn<L>, UInt<U, B>)
where
    L: IsDynGreater<UInt<U, B>>,
{
    type Out = <L as IsDynGreater<UInt<U, B>>>::Out;
}
impl<T, U> IsGreater for (T, U)
where
    T: num::IsGreater<U>,
    <T as num::IsGreater<U>>::Output: Bool,
{
    type Out = <<T as num::IsGreater<U>>::Output as Bool>::Out;
}

pub trait IsGreaterOrEqual {
    type Out;
}
impl<T, U> IsGreaterOrEqual for (T, U)
where
    (T, U): IsGreater,
    (T, U): IsEqual,
    (<(T, U) as IsGreater>::Out, <(T, U) as IsEqual>::Out): Or,
{
    type Out = <(<(T, U) as IsGreater>::Out, <(T, U) as IsEqual>::Out) as Or>::Out;
}

pub trait IsLessOrEqual {
    type Out;
}
impl<T, U> IsLessOrEqual for (T, U)
where
    (T, U): IsLess,
    (T, U): IsEqual,
    (<(T, U) as IsLess>::Out, <(T, U) as IsEqual>::Out): Or,
{
    type Out = <(<(T, U) as IsLess>::Out, <(T, U) as IsEqual>::Out) as Or>::Out;
}



================================================
FILE: src/diagnostic.rs
================================================
use crate::{False, True};

pub trait Operation {}

#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}",
    label = "Shape Mismatch"
)]
pub trait Truthy1<Op: Operation, A> {}
#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}, {B}",
    label = "Shape Mismatch"
)]
pub trait Truthy<Op: Operation, A, B> {}
#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}, {B}, {C}",
    label = "Shape Mismatch"
)]
pub trait Truthy3<Op: Operation, A, B, C> {}
#[diagnostic::on_unimplemented(
    message = "Incompatible dimensions for operation \"{Op}\": {A}, {B}, {C}, {D}",
    label = "Shape Mismatch"
)]
pub trait Truthy4<Op: Operation, A, B, C, D> {}
impl<Op: Operation, A> Truthy1<Op, A> for True {}
impl<Op: Operation, A, B> Truthy<Op, A, B> for True {}
impl<Op: Operation, A, B, C> Truthy3<Op, A, B, C> for True {}
impl<Op: Operation, A, B, C, D> Truthy4<Op, A, B, C, D> for True {}

#[allow(unused)]
#[diagnostic::on_unimplemented(
    message = "Incompatible dimension for operation \"{Op}\": {Lhs}, {Rhs}",
    label = "Shape Mismatch"
)]
pub(crate) trait Falsy<Op: Operation, Lhs, Rhs> {}
impl<Op: Operation, Lhs, Rhs> Falsy<Op, Lhs, Rhs> for False {}

#[allow(unused)]
#[diagnostic::on_unimplemented(
    message = "[glowstick shape]: {T}",
    label = "glowstick::debug_tensor!()",
    note = "This error is due to a `debug_tensor!()` macro invocation."
)]
#[allow(private_bounds)]
pub trait Diagnostic<T>: DebugTensorInvocation {
    crate::private!();
}
trait DebugTensorInvocation {
    crate::private!();
}

#[macro_export]
macro_rules! dbg_shape {
    ($t:ty) => {
        diagnostic_msg::<$t>();
    };
}
#[macro_export]
macro_rules! debug_tensor {
    ($t:ident) => {
        fn diagnostic_msg<T>(t: &T)
        where
            T: $crate::Tensor,
            T: $crate::diagnostic::Diagnostic<
                <<T as $crate::Tensor>::Shape as $crate::ShapeDiagnostic>::Out,
            >,
        {
        }
        diagnostic_msg::<_>(&$t);
    };
}



================================================
FILE: src/dynamic.rs
================================================
use std::marker::PhantomData;

use typosaurus::bool::{And, True};
use typosaurus::collections::tuple::{self, Tuplify};

use crate::DecimalDiagnostic;
use crate::cmp::{IsEqual, Max};
use crate::num::{Mul, UInt};

pub trait Dim {
    type Id;
    crate::private!();
}
impl<T> Dim for super::Dyn<T> {
    type Id = Self;
    crate::private_impl!();
}

pub trait Dynamic {
    type Label;
}

pub trait IsDynEqual<Rhs> {
    type Out;
}
pub trait IsDynGreater<Rhs> {
    type Out;
}
pub trait IsDynLess<Rhs> {
    type Out;
}
pub trait DynMax<Rhs> {
    type Out;
}
pub trait DynMin<Rhs> {
    type Out;
}
pub trait DynAdd<Rhs> {
    type Out;
}
pub trait DynMul<Rhs> {
    type Out;
}

pub struct Any;
impl Dynamic for Any {
    type Label = Any;
}
impl Tuplify for Any {
    type Out = ();
}
impl tuple::Value for Any {
    type Out = ();
    fn value() {}
}
impl<U> IsDynEqual<U> for Any {
    type Out = True;
}
impl<U> IsDynGreater<U> for Any {
    type Out = True;
}
impl<U> IsDynLess<U> for Any {
    type Out = True;
}
impl<U> DynMax<U> for Any {
    type Out = Any;
}
impl<U> DynMin<U> for Any {
    type Out = Any;
}
impl<U> DynAdd<U> for Any {
    type Out = Any;
}
impl<U> DynMul<U> for Any {
    type Out = Any;
}
pub type Wild = Any;

pub struct Term<Coeff, Var>(PhantomData<Coeff>, PhantomData<Var>);
impl<Coeff1, Var1, Coeff2, Var2> IsDynEqual<Term<Coeff2, Var2>> for Term<Coeff1, Var1>
where
    (Coeff1, Coeff2): IsEqual,
    Var1: IsDynEqual<Var2>,
    (
        <(Coeff1, Coeff2) as IsEqual>::Out,
        <Var1 as IsDynEqual<Var2>>::Out,
    ): And,
{
    type Out = <(
        <(Coeff1, Coeff2) as IsEqual>::Out,
        <Var1 as IsDynEqual<Var2>>::Out,
    ) as And>::Out;
}
impl<Coeff1, Var1, Coeff2, Var2> DynMul<Term<Coeff2, Var2>> for Term<Coeff1, Var1>
where
    (Coeff1, Coeff2): Mul,
    Var1: DynMul<Var2>,
{
    type Out = Term<<(Coeff1, Coeff2) as Mul>::Out, <Var1 as DynMul<Var2>>::Out>;
}
impl<Coeff1, Var1, Coeff2, Var2> DynMax<Term<Coeff2, Var2>> for Term<Coeff1, Var1>
where
    (Coeff1, Coeff2): Max,
    Var1: DynMax<Var2>,
{
    type Out = Term<<(Coeff1, Coeff2) as Max>::Out, <Var1 as DynMax<Var2>>::Out>;
}
impl<Coeff1, Var1, U, B> DynMax<UInt<U, B>> for Term<Coeff1, Var1> {
    type Out = Term<Coeff1, Var1>;
}
impl<Coeff, Var, U, B> IsDynEqual<UInt<U, B>> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var> IsDynEqual<Any> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var, T> IsDynGreater<T> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var, T> IsDynLess<T> for Term<Coeff, Var> {
    type Out = True;
}
impl<Coeff, Var, U, B> DynMul<UInt<U, B>> for Term<Coeff, Var>
where
    (Coeff, UInt<U, B>): Mul,
{
    type Out = Term<<(Coeff, UInt<U, B>) as Mul>::Out, Var>;
}
impl<Coeff, Var> Dynamic for Term<Coeff, Var>
where
    Coeff: DecimalDiagnostic,
{
    type Label = (<Coeff as DecimalDiagnostic>::Out, Var);
}

pub struct DynProduct<T, U>(PhantomData<T>, PhantomData<U>);
impl<T, U> IsDynEqual<DynProduct<T, U>> for DynProduct<T, U> {
    type Out = True;
}

// TODO: explore type-level map/counter for this
#[macro_export]
macro_rules! dyndims {
    {$id:ident:$label:ident} => {
        pub struct $label;
        pub type $id = $crate::Dyn<$crate::dynamic::Term<$crate::num::U1, $label>>;
        impl $crate::dynamic::IsDynEqual<$label> for $label {
            type Out = $crate::True;
        }
        impl $crate::dynamic::DynMax<$label> for $label {
            type Out = $label;
        }
    };
    {$id1:ident:$label1:ident,$($id2:ident:$label2:ident),+} => {
        dyndims!{$id1:$label1}

        $(
            impl $crate::dynamic::DynMul<$label2> for $label1 {
                type Out = $crate::dynamic::DynProduct<$label1, $label2>;
            }
            impl $crate::dynamic::DynMul<$label1> for $label2 {
                type Out = $crate::dynamic::DynProduct<$label1, $label2>;
            }
        )+

        dyndims!{$($id2:$label2),+}
    };
}



================================================
FILE: src/lib.rs
================================================
use core::marker::PhantomData;

use dynamic::Dynamic;
use typosaurus::collections::list::{All, Skippable, Takeable};
use typosaurus::num::{
    Bit, NonZero, UInt, Unsigned,
    consts::{U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10},
};
use typosaurus::{
    bool::{And, Or},
    collections::list::{Rev, Reversible},
    traits::{fold::Foldable, functor::Mapper},
};

pub use typosaurus::collections::Container;
pub use typosaurus::collections::list::{Indexed, List, Zippable};
pub use typosaurus::traits::functor::Map;

pub mod cmp;
pub mod diagnostic;
pub mod dynamic;
pub mod num;
pub mod op;
use cmp::{IsEqual, IsGreater, IsLess, Max};
use num::{Add, Div, Rem, Sub, monoid::Multiplication};
pub use typosaurus::assert_type_eq;
pub use typosaurus::bool::{False, True};
pub use typosaurus::collections::tuple;
pub use typosaurus::collections::value_list::List as ValueList;
pub use typosaurus::collections::{
    array::Arrayify,
    list::{Empty, List as Shp},
    tuple::Tuplify,
};
pub use typosaurus::list;
pub use typosaurus::traits::semigroup::Mappend;

#[macro_export]
macro_rules! shape {
    [$a:ident] => { $crate::TensorShape<$crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)>> };
    [$a:ident,$($bs:ident),+] => { $crate::TensorShape<<($crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)>, $crate::fragment![$($bs),+]) as $crate::Mappend>::Out> };
}
#[macro_export]
macro_rules! fragment {
    [$a:ident] => { $crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)> };
    [$a:ident,$($bs:ident),+] => { <($crate::Shp<(<$a as $crate::ValidDim>::Out, $crate::Empty)>, $crate::fragment![$($bs),+]) as $crate::Mappend>::Out };
}
pub type Shape1<A> = shape![A];
pub type Shape2<A, B> = shape![A, B];
pub type Shape3<A, B, C> = shape![A, B, C];
pub type Shape4<A, B, C, D> = shape![A, B, C, D];
pub type Shape5<A, B, C, D, E> = shape![A, B, C, D, E];
pub type Shape6<A, B, C, D, E, F> = shape![A, B, C, D, E, F];
pub type Shape7<A, B, C, D, E, F, G> = shape![A, B, C, D, E, F, G];
pub type Shape8<A, B, C, D, E, F, G, H> = shape![A, B, C, D, E, F, G, H];
pub type Shape9<A, B, C, D, E, F, G, H, I> = shape![A, B, C, D, E, F, G, H, I];
pub type Shape10<A, B, C, D, E, F, G, H, I, J> = shape![A, B, C, D, E, F, G, H, I, J];
pub type Shape11<A, B, C, D, E, F, G, H, I, J, K> = shape![A, B, C, D, E, F, G, H, I, J, K];
pub type Shape12<A, B, C, D, E, F, G, H, I, J, K, L> = shape![A, B, C, D, E, F, G, H, I, J, K, L];

pub(crate) struct Private;
macro_rules! private {
    () => {
        /// The trait is sealed. It was made by those who authored the crate,
        /// and the authors keep it.
        #[doc(hidden)]
        #[allow(private_interfaces)]
        fn __glowstick_private__(&self) -> crate::Private;
    };
}
macro_rules! private_impl {
    () => {
        /// The trait is sealed.
        #[allow(private_interfaces)]
        fn __glowstick_private__(&self) -> crate::Private {
            crate::Private
        }
    };
}
pub(crate) use private;
pub(crate) use private_impl;

/// A dynamic dimension which cannot be checked at compile-time.
pub struct Dyn<Label>(PhantomData<Label>);
impl<T> Tuplify for Dyn<T> {
    type Out = Dyn<T>;
}
impl<T: tuple::Value> tuple::Value for Dyn<T> {
    type Out = <T as tuple::Value>::Out;
    fn value() -> <Self as tuple::Value>::Out {
        <T as tuple::Value>::value()
    }
}

type Product<T> = <T as Foldable<Multiplication>>::Out;

pub struct IsLessThan<M>(PhantomData<M>);
impl<N, M> Mapper<N> for IsLessThan<M>
where
    (N, M): IsLess,
{
    type Out = <(N, M) as IsLess>::Out;
}
pub struct IsGreaterThan<M>(PhantomData<M>);
impl<N, M> Mapper<N> for IsGreaterThan<M>
where
    (N, M): IsGreater,
{
    type Out = <(N, M) as IsGreater>::Out;
}
type LessThan<T, N> = <(T, IsLessThan<N>) as Map<<T as Container>::Content, IsLessThan<N>>>::Out;
type AllLessThan<T, N> = All<LessThan<T, N>>;
type GreaterThan<T, N> =
    <(T, IsGreaterThan<N>) as Map<<T as Container>::Content, IsGreaterThan<N>>>::Out;
pub type AllGreaterThan<T, N> = All<GreaterThan<T, N>>;

pub struct PermutationOf<T>(PhantomData<T>);
impl<T, N> Mapper<N> for PermutationOf<T>
where
    (T, N): Dimensioned,
{
    type Out = <(T, N) as Dimensioned>::Out;
}

pub trait Tensor {
    type Shape: Shape;
}

pub trait Shape {
    type Fragment: ShapeFragment;
    type Dim<N>: Dimension
    where
        (Self, N): Dimensioned;
    type Rank: Rank;
    const RANK: usize;

    fn iter() -> Box<dyn Iterator<Item = usize>>;
    private!();
}
pub trait ValidDim {
    type Out;
    private!();
}
impl<T> ValidDim for T
where
    T: NonZero,
{
    type Out = T;
    private_impl!();
}
impl<L> ValidDim for Dyn<L> {
    type Out = Dyn<L>;
    private_impl!();
}

pub struct TensorShape<T>(T)
where
    T: ShapeFragment;

impl<T> Shape for TensorShape<T>
where
    T: ShapeFragment,
{
    type Fragment = T;
    type Dim<N>
        = <(Self, N) as Dimensioned>::Out
    where
        (Self, N): Dimensioned;
    type Rank = <T as ShapeFragment>::Rank;
    const RANK: usize = <Self::Rank as Unsigned>::USIZE;

    fn iter() -> Box<dyn Iterator<Item = usize>> {
        let n = <<T as ShapeFragment>::Dim>::USIZE;
        match n {
            0 => Box::new(std::iter::empty()),
            d => Box::new(
                std::iter::once(d)
                    .chain(<TensorShape<<T as ShapeFragment>::Next> as Shape>::iter()),
            ),
        }
    }

    private_impl!();
}
impl<T> Tuplify for TensorShape<T>
where
    T: ShapeFragment + Tuplify,
{
    type Out = <T as Tuplify>::Out;
}
impl<T> tuple::Value for TensorShape<T>
where
    T: ShapeFragment + tuple::Value,
{
    type Out = <T as tuple::Value>::Out;
    fn value() -> <Self as tuple::Value>::Out {
        <T as tuple::Value>::value()
    }
}

pub trait Dimensioned {
    type Out: Dimension;
    private!();
}
impl<T, N> Dimensioned for (TensorShape<T>, N)
where
    T: ShapeFragment,
    (T, N): Indexed,
    <(T, N) as Indexed>::Out: Dimension,
{
    type Out = <(T, N) as Indexed>::Out;
    private_impl!();
}

pub trait SkipFragment {
    type Out: ShapeFragment;
    private!();
}
impl<T> SkipFragment for (TensorShape<T>, U0)
where
    T: ShapeFragment,
{
    type Out = T;
    private_impl!();
}
impl<T, U, B> SkipFragment for (TensorShape<T>, UInt<U, B>)
where
    T: ShapeFragment,
    (T, UInt<U, B>): Skippable,
    <(T, UInt<U, B>) as Skippable>::Out: ShapeFragment,
{
    type Out = <(T, UInt<U, B>) as Skippable>::Out;
    private_impl!();
}

pub trait ZipFragment {
    type Out;
    private!();
}
impl<T, U> ZipFragment for (TensorShape<T>, TensorShape<U>)
where
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): Zippable,
{
    type Out = <(T, U) as Zippable>::Out;
    private_impl!();
}

pub trait TakeFragment {
    type Out: ShapeFragment;
    private!();
}
impl<T, N> TakeFragment for (TensorShape<T>, N)
where
    T: ShapeFragment,
    (T, N): Takeable,
    <(T, N) as Takeable>::Out: ShapeFragment,
{
    type Out = <(T, N) as Takeable>::Out;
    private_impl!();
}

pub trait IsFragEqual {
    type Out;
}
impl IsFragEqual for (Empty, Empty) {
    type Out = True;
}
impl<T, U> IsFragEqual for (Shp<(T, U)>, Empty) {
    type Out = False;
}
impl<T, U> IsFragEqual for (Empty, Shp<(T, U)>) {
    type Out = False;
}
impl<T1, U1, T2, U2> IsFragEqual for (Shp<(T1, U1)>, Shp<(T2, U2)>)
where
    (T1, T2): IsDimEqual,
    (U1, U2): IsFragEqual,
    (
        <(T1, T2) as IsDimEqual>::Out,
        <(U1, U2) as IsFragEqual>::Out,
    ): And,
{
    type Out = <(
        <(T1, T2) as IsDimEqual>::Out,
        <(U1, U2) as IsFragEqual>::Out,
    ) as And>::Out;
}

pub trait MaxDim {
    type Out: Dimension;
    private!();
}
impl<T> MaxDim for TensorShape<T>
where
    T: ShapeFragment + MaxDim,
{
    type Out = <T as MaxDim>::Out;
    private_impl!();
}
impl MaxDim for Empty {
    type Out = U0;
    private_impl!();
}
impl<T, U> MaxDim for Shp<(T, U)>
where
    U: MaxDim,
    (T, <U as MaxDim>::Out): Max,
    <(T, <U as MaxDim>::Out) as Max>::Out: Dimension,
{
    type Out = <(T, <U as MaxDim>::Out) as Max>::Out;
    private_impl!();
}

pub trait MaxDims {
    type Out;
    private!();
}
impl<T, U> MaxDims for (TensorShape<T>, TensorShape<U>)
where
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): MaxDims,
{
    type Out = <(T, U) as MaxDims>::Out;
    private_impl!();
}
impl MaxDims for (Empty, Empty) {
    type Out = Empty;
    private_impl!();
}
impl<T1, T2, U1, U2> MaxDims for (Shp<(T1, T2)>, Shp<(U1, U2)>)
where
    (T1, U1): Max,
    (T2, U2): MaxDims,
{
    type Out = Shp<(<(T1, U1) as Max>::Out, <(T2, U2) as MaxDims>::Out)>;
    private_impl!();
}

pub trait IsFragEqualOrOne {
    type Out;
    private!();
}
impl<T, U> IsFragEqualOrOne for (TensorShape<T>, TensorShape<U>)
where
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): IsFragEqualOrOne,
{
    type Out = <(T, U) as IsFragEqualOrOne>::Out;
    private_impl!();
}
impl IsFragEqualOrOne for (Empty, Empty) {
    type Out = True;
    private_impl!();
}
impl<T, U> IsFragEqualOrOne for (Shp<(T, U)>, Empty) {
    type Out = False;
    private_impl!();
}
impl<T, U> IsFragEqualOrOne for (Empty, Shp<(T, U)>) {
    type Out = False;
    private_impl!();
}
impl<T1, U1, T2, U2> IsFragEqualOrOne for (Shp<(T1, U1)>, Shp<(T2, U2)>)
where
    (T1, T2): IsDimEqualOrOne,
    (U1, U2): IsFragEqualOrOne,
    (
        <(T1, T2) as IsDimEqualOrOne>::Out,
        <(U1, U2) as IsFragEqualOrOne>::Out,
    ): And,
{
    type Out = <(
        <(T1, T2) as IsDimEqualOrOne>::Out,
        <(U1, U2) as IsFragEqualOrOne>::Out,
    ) as And>::Out;
    private_impl!();
}

pub trait ShapeFragment: Sized {
    type Dim: Dimension;
    type Rank: Rank;
    type Next: ShapeFragment;
    private!();
}
impl ShapeFragment for Empty {
    type Dim = U0;
    type Rank = U0;
    type Next = Empty;
    private_impl!();
}
impl<Dim, T> ShapeFragment for Shp<(Dim, T)>
where
    Dim: Dimension,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, U1): AddRank,
    <(<T as ShapeFragment>::Rank, U1) as AddRank>::Out: Rank,
{
    type Dim = Dim;
    type Rank = <(<T as ShapeFragment>::Rank, U1) as AddRank>::Out;
    type Next = T;
    private_impl!();
}

pub trait Rank: Unsigned {
    private!();
}
impl<U, B> Rank for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    private_impl!();
}
impl Rank for U0 {
    private_impl!();
}

pub trait IsRankEqual {
    type Out;
    private!();
}
impl<T, U> IsRankEqual for (T, U)
where
    T: Rank,
    U: Rank,
    (T, U): IsEqual,
{
    type Out = <(T, U) as IsEqual>::Out;
    private_impl!();
}

pub trait AddRank {
    type Out: Rank;
    private!();
}
impl<R1> AddRank for (R1, U1)
where
    R1: Rank,
    (R1, U1): Add,
    <(R1, U1) as Add>::Out: Rank,
{
    type Out = <(R1, U1) as Add>::Out;
    private_impl!();
}

pub trait Dimension {
    const USIZE: usize;
    private!();
}
impl<U, B> Dimension for UInt<U, B>
where
    U: Unsigned,
    B: Bit,
{
    const USIZE: usize = <UInt<U, B> as Unsigned>::USIZE;
    private_impl!();
}
impl Dimension for U0 {
    const USIZE: usize = 0;
    private_impl!();
}
impl<L> Dimension for Dyn<L> {
    const USIZE: usize = 0; // TODO: what should this be?
    private_impl!();
}
pub trait IsDimEqual {
    type Out;
    private!();
}
impl<T, U> IsDimEqual for (T, U)
where
    T: Dimension,
    U: Dimension,
    (T, U): IsEqual,
{
    type Out = <(T, U) as IsEqual>::Out;
    private_impl!();
}

pub trait IsDimEqualOrOne {
    type Out;
    private!();
}
impl<T, U> IsDimEqualOrOne for (T, U)
where
    T: Dimension,
    U: Dimension,
    (T, U): IsEqual,
    (T, U1): IsEqual,
    (U, U1): IsEqual,
    (<(T, U) as IsEqual>::Out, <(U, U1) as IsEqual>::Out): Or,
    (
        <(<(T, U) as IsEqual>::Out, <(U, U1) as IsEqual>::Out) as Or>::Out,
        <(T, U1) as IsEqual>::Out,
    ): Or,
{
    type Out = <(
        <(<(T, U) as IsEqual>::Out, <(U, U1) as IsEqual>::Out) as Or>::Out,
        <(T, U1) as IsEqual>::Out,
    ) as Or>::Out;
    private_impl!();
}

// Diagnostic labels
pub struct IDX<T>(PhantomData<T>);
pub struct RANK<T>(PhantomData<T>);
pub struct _0;
pub struct _1;
pub struct _2;
pub struct _3;
pub struct _4;
pub struct _5;
pub struct _6;
pub struct _7;
pub struct _8;
pub struct _9;

impl Tuplify for _0 {
    type Out = _0;
}
impl Tuplify for _1 {
    type Out = _1;
}
impl Tuplify for _2 {
    type Out = _2;
}
impl Tuplify for _3 {
    type Out = _3;
}
impl Tuplify for _4 {
    type Out = _4;
}
impl Tuplify for _5 {
    type Out = _5;
}
impl Tuplify for _6 {
    type Out = _6;
}
impl Tuplify for _7 {
    type Out = _7;
}
impl Tuplify for _8 {
    type Out = _8;
}
impl Tuplify for _9 {
    type Out = _9;
}

pub trait DimensionDiagnostic {
    type Out;
    private!();
}
impl<T> DimensionDiagnostic for (T, U0) {
    type Out = Shp<()>;
    private_impl!();
}
impl<T, U, B> DimensionDiagnostic for (T, UInt<U, B>)
where
    T: ShapeFragment,
    (<T as ShapeFragment>::Dim, U10): Div,
    (<T as ShapeFragment>::Dim, U10): Rem,
    <(<T as ShapeFragment>::Dim, U10) as Rem>::Out: DecimalDiagnostic,
    (<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0): IsEqual,
    (
        Dec<<T as ShapeFragment>::Dim>,
        <(<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0) as IsEqual>::Out,
    ): DecimalDiagnostic,
    <(
        Dec<<T as ShapeFragment>::Dim>,
        <(<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0) as IsEqual>::Out,
    ) as DecimalDiagnostic>::Out: Reversible,
    (UInt<U, B>, U1): Sub,
    (NextFrag<T>, <(UInt<U, B>, U1) as Sub>::Out): DimensionDiagnostic,
{
    type Out = Shp<(
        DIM<
            Rev<
                <(
                    Dec<<T as ShapeFragment>::Dim>,
                    <(<(<T as ShapeFragment>::Dim, U10) as Div>::Out, U0) as IsEqual>::Out,
                ) as DecimalDiagnostic>::Out,
            >,
        >,
        <(NextFrag<T>, <(UInt<U, B>, U1) as Sub>::Out) as DimensionDiagnostic>::Out,
    )>;
    private_impl!();
}

type NextFrag<T> = <T as ShapeFragment>::Next;
pub trait ShapeDiagnostic {
    type Out;
    private!();
}
impl<T> ShapeDiagnostic for TensorShape<T>
where
    T: ShapeFragment,
    <T as ShapeFragment>::Rank: RankDiagnostic<T>,
{
    type Out = <<T as ShapeFragment>::Rank as RankDiagnostic<T>>::Out;
    private_impl!();
}
pub trait RankDiagnostic<T> {
    type Out;
    private!();
}

impl<T, N> RankDiagnostic<T> for N
where
    (N, U10): Div,
    (<(N, U10) as Div>::Out, U0): IsEqual,
    (Dec<N>, <(<(N, U10) as Div>::Out, U0) as IsEqual>::Out): DecimalDiagnostic,
    <(Dec<N>, <(<(N, U10) as Div>::Out, U0) as IsEqual>::Out) as DecimalDiagnostic>::Out: Tuplify,
    (T, N): DimensionDiagnostic,
    <(T, N) as DimensionDiagnostic>::Out: Tuplify,
{
    type Out = (
        RANK<<<(Dec<N>, <(<(N, U10) as Div>::Out, U0) as IsEqual>::Out) as DecimalDiagnostic>::Out as Tuplify>::Out>,
        <<(T, N) as DimensionDiagnostic>::Out as Tuplify>::Out,
    );
    private_impl!();
}

pub struct DIM<T>(PhantomData<T>);
impl<T: Tuplify> Tuplify for DIM<T> {
    type Out = DIM<<T as Tuplify>::Out>;
}
pub trait DecimalDiagnostic {
    type Out;
    private!();
}
macro_rules! decimpl {
    [($n:ident,$t:ident)] => {
        impl DecimalDiagnostic for $n {
            type Out = $t;
            private_impl!();
        }
    };
    [($n:ident,$t:ident),$(($ns:ident,$ts:ident)),+] => { decimpl![($n,$t)]; decimpl![$(($ns,$ts)),+]; };
}
decimpl![
    (U0, _0),
    (U1, _1),
    (U2, _2),
    (U3, _3),
    (U4, _4),
    (U5, _5),
    (U6, _6),
    (U7, _7),
    (U8, _8),
    (U9, _9)
];
pub struct Dec<T>(PhantomData<T>);
impl<L> DecimalDiagnostic for Dyn<L>
where
    L: Dynamic,
{
    type Out = Dyn<<L as Dynamic>::Label>;
    private_impl!();
}
impl<T> DecimalDiagnostic for (Dec<T>, True)
where
    (T, U10): Rem,
    <(T, U10) as Rem>::Out: DecimalDiagnostic,
{
    type Out = Shp<(<<(T, U10) as Rem>::Out as DecimalDiagnostic>::Out, Shp<()>)>;
    private_impl!();
}
impl<T> DecimalDiagnostic for (Dec<T>, False)
where
    (T, U10): Div,
    (T, U10): Rem,
    <(T, U10) as Rem>::Out: DecimalDiagnostic,
    (<(<(T, U10) as Div>::Out, U10) as Div>::Out, U0): IsEqual,
    (<(T, U10) as Div>::Out, U10): Div,
    (
        Dec<<(T, U10) as Div>::Out>,
        <(<(<(T, U10) as Div>::Out, U10) as Div>::Out, U0) as IsEqual>::Out,
    ): DecimalDiagnostic,
    (Dec<T>, True): DecimalDiagnostic,
{
    type Out = Shp<(
        <<(T, U10) as Rem>::Out as DecimalDiagnostic>::Out,
        <(
            Dec<<(T, U10) as Div>::Out>,
            <(<(<(T, U10) as Div>::Out, U10) as Div>::Out, U0) as IsEqual>::Out,
        ) as DecimalDiagnostic>::Out,
    )>;
    private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        num::consts::{U0, U2, U3},
    };

    use super::*;

    #[allow(unused)]
    #[test]
    fn dims() {
        type MyShape = shape![U1, U2, U3];
        assert_type_eq!(<MyShape as Shape>::Dim<U0>, U1);
        assert_type_eq!(<MyShape as Shape>::Dim<U1>, U2);
        assert_type_eq!(<MyShape as Shape>::Dim<U2>, U3);
    }

    #[allow(unused)]
    #[test]
    fn dim_eq() {
        type F = <(U0, U1) as IsDimEqual>::Out;
        assert_type_eq!(F, False);

        type T = <(U1, U1) as IsDimEqual>::Out;
        assert_type_eq!(T, True);
    }

    #[allow(unused)]
    #[test]
    fn frag_eq() {
        type La1 = fragment![U1, U1, U2];
        type Lb1 = fragment![U1, U1, U1];
        assert_type_eq!(<(La1, Lb1) as IsFragEqual>::Out, False);

        type La2 = fragment![U1, U1, U2];
        type Lb2 = fragment![U1, U1, U2];
        type Fe = <(La2, Lb2) as IsFragEqual>::Out;
        assert_type_eq!(Fe, True);

        assert_type_eq!(<(Empty, Empty) as IsFragEqual>::Out, True);
        assert_type_eq!(<(fragment![U1], Empty) as IsFragEqual>::Out, False);
        assert_type_eq!(<(Empty, fragment![U1]) as IsFragEqual>::Out, False);
        assert_type_eq!(<(fragment![U1], fragment![U1]) as IsFragEqual>::Out, True);
    }

    #[allow(unused)]
    #[test]
    fn dyn_diag() {
        struct BatchSize;
        impl Dynamic for BatchSize {
            type Label = Self;
        }
        type B = Dyn<BatchSize>;
        type DynShape = shape![U1, U1, B];
        type Diag = <DynShape as ShapeDiagnostic>::Out;
        assert_type_eq!(Diag, (RANK<_3>, (DIM<_1>, DIM<_1>, DIM<Dyn<BatchSize>>)));
    }
}



================================================
FILE: src/num.rs
================================================
use crate::Dyn;
use crate::dynamic::DynAdd;
use crate::dynamic::DynMul;
use typosaurus::collections::list::{Empty, List};
use typosaurus::num::UTerm;

pub use typosaurus::num::consts::*;
pub use typosaurus::num::{UInt, Unsigned};
use typosaurus::traits::functor::Mapper;

pub trait Div {
    type Out;
}
impl<U, B, L> Div for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Div for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Div for (T, U)
where
    T: core::ops::Div<U>,
{
    type Out = <T as core::ops::Div<U>>::Output;
}

pub trait Rem {
    type Out;
}
impl<U, B, L> Rem for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Rem for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Rem for (T, U)
where
    T: core::ops::Rem<U>,
{
    type Out = <T as core::ops::Rem<U>>::Output;
}

pub trait Add {
    type Out;
}
impl<L> Add for (UTerm, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<L> Add for (Dyn<L>, UTerm) {
    type Out = Dyn<L>;
}
impl<U, B, L> Add for (UInt<U, B>, Dyn<L>)
where
    L: DynAdd<UInt<U, B>>,
{
    type Out = Dyn<<L as DynAdd<UInt<U, B>>>::Out>;
}
impl<U, B, L> Add for (Dyn<L>, UInt<U, B>)
where
    L: DynAdd<UInt<U, B>>,
{
    type Out = Dyn<<L as DynAdd<UInt<U, B>>>::Out>;
}
impl<T, U> Add for (Dyn<T>, Dyn<U>)
where
    T: DynAdd<U>,
{
    type Out = Dyn<<T as DynAdd<U>>::Out>;
}
impl<T, U> Add for (T, U)
where
    T: core::ops::Add<U>,
{
    type Out = <T as core::ops::Add<U>>::Output;
}

pub trait Sub {
    type Out;
}
impl<L> Sub for (UTerm, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<L> Sub for (Dyn<L>, UTerm) {
    type Out = Dyn<L>;
}
impl<U, B, L> Sub for (UInt<U, B>, Dyn<L>) {
    type Out = Dyn<L>;
}
impl<U, B, L> Sub for (Dyn<L>, UInt<U, B>) {
    type Out = Dyn<L>;
}
impl<T, U> Sub for (T, U)
where
    T: core::ops::Sub<U>,
{
    type Out = <T as core::ops::Sub<U>>::Output;
}

pub trait Mul {
    type Out;
}
impl<L> Mul for (UTerm, Dyn<L>) {
    type Out = UTerm;
}
impl<L> Mul for (Dyn<L>, UTerm) {
    type Out = UTerm;
}
impl<U, B, L> Mul for (UInt<U, B>, Dyn<L>)
where
    L: DynMul<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMul<UInt<U, B>>>::Out>;
}
impl<U, B, L> Mul for (Dyn<L>, UInt<U, B>)
where
    L: DynMul<UInt<U, B>>,
{
    type Out = Dyn<<L as DynMul<UInt<U, B>>>::Out>;
}
impl<T, U> Mul for (Dyn<T>, Dyn<U>)
where
    T: DynMul<U>,
{
    type Out = Dyn<<T as DynMul<U>>::Out>;
}
impl<T, U> Mul for (T, U)
where
    T: core::ops::Mul<U>,
{
    type Out = <T as core::ops::Mul<U>>::Output;
}

pub struct ZipSub;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipSub
where
    (T, U): Sub,
{
    type Out = <(T, U) as Sub>::Out;
}
pub struct ZipAdd;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipAdd
where
    (T, U): Add,
{
    type Out = <(T, U) as Add>::Out;
}
pub struct ZipDiv;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipDiv
where
    (T, U): Div,
{
    type Out = <(T, U) as Div>::Out;
}
pub struct ZipDivAddOne;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipDivAddOne
where
    (T, U): Div,
    (<(T, U) as Div>::Out, U1): Add,
{
    type Out = <(<(T, U) as Div>::Out, U1) as Add>::Out;
}

// This is tailored to keff calc
pub struct ZipSubOneMul;
impl<T, U> Mapper<List<(T, List<(U, Empty)>)>> for ZipSubOneMul
where
    (T, U1): Sub,
    (U, U1): Sub,
    (<(T, U1) as Sub>::Out, <(U, U1) as Sub>::Out): Mul,
    (
        <(<(T, U1) as Sub>::Out, <(U, U1) as Sub>::Out) as Mul>::Out,
        U,
    ): Add,
{
    type Out = <(
        <(<(T, U1) as Sub>::Out, <(U, U1) as Sub>::Out) as Mul>::Out,
        U,
    ) as Add>::Out;
}

pub mod monoid {
    use typosaurus::num::consts::{U0, U1};
    use typosaurus::traits::{monoid::Mempty, semigroup::Semigroup};

    use super::*;

    pub struct Addition;
    pub struct Multiplication;

    impl<Lhs, Rhs> Semigroup<Lhs, Rhs> for Addition
    where
        (Lhs, Rhs): Add,
    {
        type Mappend = <(Lhs, Rhs) as Add>::Out;
    }
    impl Mempty for Addition {
        type Out = U0;
    }

    impl<Lhs, Rhs> Semigroup<Lhs, Rhs> for Multiplication
    where
        (Lhs, Rhs): Mul,
    {
        type Mappend = <(Lhs, Rhs) as Mul>::Out;
    }
    impl Mempty for Multiplication {
        type Out = U1;
    }
}



================================================
FILE: src/op/broadcast.rs
================================================
use typosaurus::traits::semigroup::Mappend;

use crate::{
    cmp::IsGreaterOrEqual,
    diagnostic::{self, Truthy},
    num::Sub,
    IsFragEqualOrOne, MaxDims, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
};

pub struct Broadcast;
impl diagnostic::Operation for Broadcast {}

/// Boolean type operator for `Broadcast` compatibility.
///
/// If shape `U` may be expanded to shape `T`, then the `Out`
/// associated type of this trait for `(T, U) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): IsGreaterOrEqual,
    <(
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ) as IsGreaterOrEqual>::Out: Truthy<
        Broadcast,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        <TensorShape<U> as crate::ShapeDiagnostic>::Out,
    >,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): Sub,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): SkipFragment,
    <(
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ) as SkipFragment>::Out: ShapeFragment,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ): IsFragEqualOrOne,
{
    type Out = <(
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ) as IsFragEqualOrOne>::Out;
    crate::private_impl!();
}

/// Type operator for broadcast-compatible shapes.
///
/// If shape `U` may be broadcast as shape `T`, then the
/// `Out` assocatied type of this trait for `(T, U)` is
/// the resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U> Compatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): IsGreaterOrEqual,
    <(
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ) as IsGreaterOrEqual>::Out: Truthy<
        Broadcast,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        <TensorShape<U> as crate::ShapeDiagnostic>::Out,
    >,
    (
        <TensorShape<T> as Shape>::Rank,
        <TensorShape<U> as Shape>::Rank,
    ): Sub,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): TakeFragment,
    <(
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ) as TakeFragment>::Out: ShapeFragment,
    (
        TensorShape<T>,
        <(
            <TensorShape<T> as Shape>::Rank,
            <TensorShape<U> as Shape>::Rank,
        ) as Sub>::Out,
    ): SkipFragment,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ): IsFragEqualOrOne,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out: Truthy<
        Broadcast,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        <TensorShape<U> as crate::ShapeDiagnostic>::Out,
    >,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        U,
    ): MaxDims,
    (
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        <(
            <(
                TensorShape<T>,
                <(
                    <TensorShape<T> as Shape>::Rank,
                    <TensorShape<U> as Shape>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            U,
        ) as MaxDims>::Out,
    ): Mappend,
    <(
        <(
            TensorShape<T>,
            <(
                <TensorShape<T> as Shape>::Rank,
                <TensorShape<U> as Shape>::Rank,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        <(
            <(
                TensorShape<T>,
                <(
                    <TensorShape<T> as Shape>::Rank,
                    <TensorShape<U> as Shape>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            U,
        ) as MaxDims>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                TensorShape<T>,
                <(
                    <TensorShape<T> as Shape>::Rank,
                    <TensorShape<U> as Shape>::Rank,
                ) as Sub>::Out,
            ) as TakeFragment>::Out,
            <(
                <(
                    TensorShape<T>,
                    <(
                        <TensorShape<T> as Shape>::Rank,
                        <TensorShape<U> as Shape>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                U,
            ) as MaxDims>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use core::marker::PhantomData;

    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U128, U16, U42, U422},
    };

    use super::*;

    use crate::{dynamic::Any, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn single() {
        type Rhs = shape![U1];
        type Lhs = shape![U1];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, True);
        assert_type_eq!(<(Lhs, Rhs) as Compatible>::Out, shape![U1]);
    }

    #[allow(unused)]
    #[test]
    fn ones() {
        type Rhs = shape![U1, U1, U1, U42];
        type Lhs = shape![U42, U42, U42, U1];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, True);
        assert_type_eq!(<(Lhs, Rhs) as Compatible>::Out, shape![U42, U42, U42, U42]);
    }

    #[allow(unused)]
    #[test]
    fn smaller() {
        type Rhs = shape![U1, U1, U42];
        type Lhs = shape![U42, U42, U42, U42, U42, U42, U42, U42];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, True);
    }

    #[allow(unused)]
    #[test]
    fn incompat() {
        type Rhs = shape![U1, U422, U42];
        type Lhs = shape![U42, U42, U42, U42, U42, U42, U42, U42];
        assert_type_eq!(<(Lhs, Rhs) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type Rhs = shape![U1, U1, U1, B];
        type Lhs = shape![U42, U42, U42, U1];

        type OutShape = <(Rhs, Lhs) as Compatible>::Out;
        assert_type_eq!(OutShape, shape![U42, U42, U42, B]);

        fn compat<A, B>() -> PhantomData<<(A, B) as Compatible>::Out>
        where
            (A, B): Compatible,
        {
            PhantomData
        }
        compat::<Lhs, Rhs>();
    }

    #[allow(unused)]
    #[test]
    fn op() {
        use core::marker::PhantomData;
        type B = Dyn<Any>;
        {
            struct Op<'a, T>(&'a T);
            impl<S> Op<'_, PhantomData<S>>
            where
                S: crate::Shape,
                (crate::Shape4<U1, U16, B, U128>, S): Compatible,
            {
                #[allow(clippy::type_complexity)]
                pub fn broadcast_as(
                    &self,
                ) -> PhantomData<<(crate::Shape4<U1, U16, B, U128>, S) as Compatible>::Out>
                {
                    PhantomData
                }
            }
            Op(&PhantomData::<crate::Shape4<U1, U1, U1, U1>>).broadcast_as();
        }
    }
}



================================================
FILE: src/op/cat.rs
================================================
use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    Dimensioned, IsFragEqual, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
    cmp::{IsEqual, IsGreater},
    diagnostic::{self, Truthy},
    num::Add,
};

struct Cat;
impl diagnostic::Operation for Cat {}

/// Boolean type operator for `Cat` compatibility.
///
/// If shape `U` may be concatenated with shape `T` on dimension `I`, then
/// the `Out` associated type of this trait for `(T, U, I) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U, I> IsCompatible for (TensorShape<T>, TensorShape<U>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank): IsEqual,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<U>, I): TakeFragment,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        <(TensorShape<U>, I) as TakeFragment>::Out,
    ): IsFragEqual,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<U>, <(I, U1) as Add>::Out): SkipFragment,
    (
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        <(TensorShape<U>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): IsFragEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
    ): And,
    (
        <(
            <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
            <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            <(TensorShape<U>, I) as TakeFragment>::Out,
        ) as IsFragEqual>::Out,
    ): And,
    (
        <(
            <(
                <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
                <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                <(TensorShape<U>, I) as TakeFragment>::Out,
            ) as IsFragEqual>::Out,
        ) as And>::Out,
        <(
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
            <(TensorShape<U>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as IsFragEqual>::Out,
    ): And,
{
    type Out = <(
        <(
            <(
                <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
                <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                <(TensorShape<U>, I) as TakeFragment>::Out,
            ) as IsFragEqual>::Out,
        ) as And>::Out,
        <(
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
            <(TensorShape<U>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as IsFragEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for concat-compatible shapes.
///
/// If shape `U` may be concatenated with shape `T` at dimension `I`,
/// then the `Out` assocatied type of this trait for `(T, U, I)` is
/// the resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U, I> Compatible for (TensorShape<T>, TensorShape<U>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, TensorShape<U>, I): IsCompatible,
    <(TensorShape<T>, TensorShape<U>, I) as IsCompatible>::Out:
        Truthy<Cat, TensorShape<T>, TensorShape<U>>,
    (TensorShape<T>, I): Dimensioned,
    (TensorShape<U>, I): Dimensioned,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(TensorShape<U>, I) as Dimensioned>::Out,
    ): Add,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(
            <(
                <(TensorShape<T>, I) as Dimensioned>::Out,
                <(TensorShape<U>, I) as Dimensioned>::Out,
            ) as Add>::Out,
            Empty,
        )>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(
                <(
                    <(TensorShape<T>, I) as Dimensioned>::Out,
                    <(TensorShape<U>, I) as Dimensioned>::Out,
                ) as Add>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(
                <(
                    <(TensorShape<T>, I) as Dimensioned>::Out,
                    <(TensorShape<U>, I) as Dimensioned>::Out,
                ) as Add>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(
                    <(
                        <(TensorShape<T>, I) as Dimensioned>::Out,
                        <(TensorShape<U>, I) as Dimensioned>::Out,
                    ) as Add>::Out,
                    Empty,
                )>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U2, U3, U4, U6, U42, U44},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U3, U2], U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, shape![U3, U2], U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, shape![U3, U42], U1) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U3, U2], U0) as Compatible>::Out,
            shape![U6, U2]
        );
        assert_type_eq!(
            <(MyShape, shape![U3, U2], U1) as Compatible>::Out,
            shape![U3, U4]
        );
        assert_type_eq!(
            <(MyShape, shape![U3, U42], U1) as Compatible>::Out,
            shape![U3, U44]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type ShapeA = shape![U1, U1, B];
        type ShapeB = shape![U1, U2, B];

        type OutShape = <(ShapeA, ShapeB, U1) as Compatible>::Out;
        assert_type_eq!(OutShape, shape![U1, U3, B]);
    }
}



================================================
FILE: src/op/cat_dyn.rs
================================================
use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    Dimensioned, Dyn, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::Add,
};

struct Cat;
impl diagnostic::Operation for Cat {}

/// Boolean type operator for `Cat` compatibility.
///
/// If shape `U` may be concatenated with shape `T` on dimension `I`, then
/// the `Out` associated type of this trait for `(T, U, I) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I> IsCompatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out;
    crate::private_impl!();
}

/// Type operator for concat-compatible shapes.
///
/// If shape `U` may be concatenated with shape `T` at dimension `I`,
/// then the `Out` assocatied type of this trait for `(T, U, I)` is
/// the resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, D> Compatible for (TensorShape<T>, I, Dyn<D>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    <(TensorShape<T>, I) as IsCompatible>::Out: Truthy<Cat, TensorShape<T>, I>,
    (TensorShape<T>, I): Dimensioned,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(Dyn<D>, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(Dyn<D>, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(Dyn<D>, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(Dyn<D>, Empty)>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U2, U3},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type N = Dyn<Any>;
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, N) as Compatible>::Out, shape![N, U2]);
        assert_type_eq!(<(MyShape, U1, N) as Compatible>::Out, shape![U3, N]);
    }
}



================================================
FILE: src/op/convolution.rs
================================================
use typosaurus::bool::And;
use typosaurus::bool::monoid::Both;
use typosaurus::collections::{
    Container,
    list::{Empty, List, Zippable},
};
use typosaurus::num::consts::{U0, U1, U2};
use typosaurus::traits::fold::Foldable;
use typosaurus::traits::functor::Map;
use typosaurus::traits::semigroup::Mappend;

use crate::cmp::{IsEqual, IsGreaterOrEqual};
use crate::num::{ZipAdd, ZipDivAddOne, ZipSub, ZipSubOneMul};
use crate::{
    AllGreaterThan, Dimension, Dimensioned, Shape, ShapeFragment, TensorShape,
    diagnostic::{self, Truthy},
    num::Sub,
};
use crate::{IsGreaterThan, ShapeDiagnostic, SkipFragment, TakeFragment};

pub trait Kernel<D>: Sized + Shape {
    type M: Dimension;
    type C: Dimension;
    type Sp: ShapeFragment;
    type DilateZipped;
    crate::private!();
}
impl<T, D> Kernel<D> for T
where
    T: Shape,
    D: ShapeFragment,
    (T, U2): SkipFragment,
    (T, U1): Dimensioned,
    (T, U0): Dimensioned,
    (D, <(T, U2) as SkipFragment>::Out): Zippable,
{
    type M = <(T, U0) as Dimensioned>::Out;
    type C = <(T, U1) as Dimensioned>::Out;
    type Sp = <(T, U2) as SkipFragment>::Out;
    type DilateZipped = <(D, <(T, U2) as SkipFragment>::Out) as Zippable>::Out;
    crate::private_impl!();
}

/// Boolean type operator for `Convolution` compatibility.
///
/// If shapes `T`, `K`, `P`, `S`, and `D` are compatible with convolution,
/// then the `Out` associated type of this trait for `(T, K, P, S, D)` is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, K, P1, P2, S, D> IsCompatible for (T, K, P1, P2, S, D)
where
    T: Shape,
    D: ShapeFragment + Container,
    (D, IsGreaterThan<U0>): Map<<D as Container>::Content, IsGreaterThan<U0>>,
    <(D, IsGreaterThan<U0>) as Map<<D as Container>::Content, IsGreaterThan<U0>>>::Out:
        Foldable<Both>,
    AllGreaterThan<D, U0>: typosaurus::bool::Truthy,
    S: Container,
    (S, IsGreaterThan<U0>): Map<<S as Container>::Content, IsGreaterThan<U0>>,
    <(S, IsGreaterThan<U0>) as Map<<S as Container>::Content, IsGreaterThan<U0>>>::Out:
        Foldable<Both>,
    AllGreaterThan<S, U0>: typosaurus::bool::Truthy,
    (T, U2): SkipFragment,
    (T, U1): Dimensioned,
    (T, U0): Dimensioned,
    (<(T, U2) as SkipFragment>::Out, D): Zippable,
    <(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out: Container,
    (
        <(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out,
        ZipSubOneMul,
    ): Map<
            <<(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out as Container>::Content,
            ZipSubOneMul,
        >,
    <(
        <(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out,
        ZipSubOneMul,
    ) as Map<
        <<(<(T, U2) as SkipFragment>::Out, D) as Zippable>::Out as Container>::Content,
        ZipSubOneMul,
    >>::Out: ShapeFragment,
    T: Shape,
    K: Kernel<D>,
    P1: ShapeFragment,
    P2: ShapeFragment,
    S: ShapeFragment,
    (<T as Shape>::Rank, <K as Shape>::Rank): IsGreaterOrEqual,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): Dimensioned,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as Dimensioned>::Out,
        <K as Kernel<D>>::C,
    ): IsEqual,
    (
        <P1 as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <S as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <P2 as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <D as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
        <(
            <P1 as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
    ): And,
    (
        <(
            <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
            <(
                <P1 as ShapeFragment>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <S as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
    ): And,
    (
        <(
            <(
                <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                <(
                    <P1 as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <S as ShapeFragment>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <P2 as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
    ): And,
    (
        <D as ShapeFragment>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): IsEqual,
    (
        <(
            <(
                <(
                    <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                    <(
                        <P1 as ShapeFragment>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as IsEqual>::Out,
                ) as And>::Out,
                <(
                    <S as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <P2 as ShapeFragment>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as IsEqual>::Out,
        ) as And>::Out,
        <(
            <(
                T,
                <(
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                    U1,
                ) as Sub>::Out,
            ) as Dimensioned>::Out,
            <K as Kernel<D>>::C,
        ) as IsEqual>::Out,
    ): And,
    (
        <(
            <D as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
        <(
            <(
                <(
                    <(
                        <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                        <(
                            <P1 as ShapeFragment>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as IsEqual>::Out,
                    ) as And>::Out,
                    <(
                        <S as ShapeFragment>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as IsEqual>::Out,
                ) as And>::Out,
                <(
                    <P2 as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(
                    T,
                    <(
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                        U1,
                    ) as Sub>::Out,
                ) as Dimensioned>::Out,
                <K as Kernel<D>>::C,
            ) as IsEqual>::Out,
        ) as And>::Out,
    ): And,
{
    type Out = <(
        <(
            <D as ShapeFragment>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as IsEqual>::Out,
        <(
            <(
                <(
                    <(
                        <(<T as Shape>::Rank, <K as Shape>::Rank) as IsGreaterOrEqual>::Out,
                        <(
                            <P1 as ShapeFragment>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as IsEqual>::Out,
                    ) as And>::Out,
                    <(
                        <S as ShapeFragment>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as IsEqual>::Out,
                ) as And>::Out,
                <(
                    <P2 as ShapeFragment>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as IsEqual>::Out,
            ) as And>::Out,
            <(
                <(
                    T,
                    <(
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                        U1,
                    ) as Sub>::Out,
                ) as Dimensioned>::Out,
                <K as Kernel<D>>::C,
            ) as IsEqual>::Out,
        ) as And>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Convolution`-compatibile shapes.
///
/// If shapes `T`, `K`, `P`, `S` and `D` may be used together for convolution,
/// then the `Out` associated type of this trait for `(T, K, P, S, D)` is the
/// resulting shape after convolution.
struct Convolution;
impl diagnostic::Operation for Convolution {}

pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, K, P1, P2, S, D> Compatible for (T, K, P1, P2, S, D)
where
    T: Shape + ShapeDiagnostic,
    D: ShapeFragment,
    K: Kernel<D> + ShapeDiagnostic,
    P1: ShapeFragment,
    P2: ShapeFragment,
    S: ShapeFragment,
    (T, K, P1, P2, S, D): IsCompatible,
    <(T, K, P1, P2, S, D) as IsCompatible>::Out:
        Truthy<Convolution, <T as ShapeDiagnostic>::Out, <K as ShapeDiagnostic>::Out>,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    <K as Kernel<D>>::DilateZipped: Container,
    (<K as Kernel<D>>::DilateZipped, ZipSubOneMul):
        Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>,
    //
    (
        T,
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
    ): SkipFragment,
    (
        <(
            T,
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
            <<K as Kernel<D>>::DilateZipped as Container>::Content,
            ZipSubOneMul,
        >>::Out,
    ): Zippable,
    <(
        <(
            T,
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
        ) as SkipFragment>::Out,
        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
            <<K as Kernel<D>>::DilateZipped as Container>::Content,
            ZipSubOneMul,
        >>::Out,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                T,
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                ZipSubOneMul,
            >>::Out,
        ) as Zippable>::Out,
        ZipSub,
    ): Map<
            <<(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out as Container>::Content,
            ZipSub,
        >,
    <(
        <(
            <(
                T,
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                ZipSubOneMul,
            >>::Out,
        ) as Zippable>::Out,
        ZipSub,
    ) as Map<
        <<(
            <(
                T,
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
            ) as SkipFragment>::Out,
            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                ZipSubOneMul,
            >>::Out,
        ) as Zippable>::Out as Container>::Content,
        ZipSub,
    >>::Out: ShapeFragment,
    //
    (
        <(
            <(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out,
            ZipSub,
        ) as Map<
            <<(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out as Container>::Content,
            ZipSub,
        >>::Out,
        P1,
    ): Zippable,
    <(
        <(
            <(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out,
            ZipSub,
        ) as Map<
            <<(
                <(
                    T,
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                ) as SkipFragment>::Out,
                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                    ZipSubOneMul,
                >>::Out,
            ) as Zippable>::Out as Container>::Content,
            ZipSub,
        >>::Out,
        P1,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                <(
                    <(
                        T,
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                    ) as SkipFragment>::Out,
                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                        ZipSubOneMul,
                    >>::Out,
                ) as Zippable>::Out,
                ZipSub,
            ) as Map<
                <<(
                    <(
                        T,
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                    ) as SkipFragment>::Out,
                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                        ZipSubOneMul,
                    >>::Out,
                ) as Zippable>::Out as Container>::Content,
                ZipSub,
            >>::Out,
            P1,
        ) as Zippable>::Out,
        ZipAdd,
    ): Map<
            <<(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >,
    //
    (
        <(
            <(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        P2,
    ): Zippable,
    <(
        <(
            <(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out,
                    ZipSub,
                ) as Map<
                    <<(
                        <(
                            T,
                            <(
                                <T as Shape>::Rank,
                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                            ) as Sub>::Out,
                        ) as SkipFragment>::Out,
                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                            ZipSubOneMul,
                        >>::Out,
                    ) as Zippable>::Out as Container>::Content,
                    ZipSub,
                >>::Out,
                P1,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        P2,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out,
                        ZipSub,
                    ) as Map<
                        <<(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out as Container>::Content,
                        ZipSub,
                    >>::Out,
                    P1,
                ) as Zippable>::Out,
                ZipAdd,
            ) as Map<
                <<(
                    <(
                        <(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out,
                        ZipSub,
                    ) as Map<
                        <<(
                            <(
                                T,
                                <(
                                    <T as Shape>::Rank,
                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                ) as Sub>::Out,
                            ) as SkipFragment>::Out,
                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                ZipSubOneMul,
                            >>::Out,
                        ) as Zippable>::Out as Container>::Content,
                        ZipSub,
                    >>::Out,
                    P1,
                ) as Zippable>::Out as Container>::Content,
                ZipAdd,
            >>::Out,
            P2,
        ) as Zippable>::Out,
        ZipAdd,
    ): Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >,
    //
    (
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        S,
    ): Zippable,
    <(
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out,
            ZipAdd,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out,
                            ZipSub,
                        ) as Map<
                            <<(
                                <(
                                    T,
                                    <(
                                        <T as Shape>::Rank,
                                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                    ) as Sub>::Out,
                                ) as SkipFragment>::Out,
                                <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                    <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                    ZipSubOneMul,
                                >>::Out,
                            ) as Zippable>::Out as Container>::Content,
                            ZipSub,
                        >>::Out,
                        P1,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                P2,
            ) as Zippable>::Out as Container>::Content,
            ZipAdd,
        >>::Out,
        S,
    ) as Zippable>::Out: Container,
    (
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    P2,
                ) as Zippable>::Out,
                ZipAdd,
            ) as Map<
                <<(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out,
                                ZipSub,
                            ) as Map<
                                <<(
                                    <(
                                        T,
                                        <(
                                            <T as Shape>::Rank,
                                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                        ) as Sub>::Out,
                                    ) as SkipFragment>::Out,
                                    <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                        <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                        ZipSubOneMul,
                                    >>::Out,
                                ) as Zippable>::Out as Container>::Content,
                                ZipSub,
                            >>::Out,
                            P1,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    P2,
                ) as Zippable>::Out as Container>::Content,
                ZipAdd,
            >>::Out,
            S,
        ) as Zippable>::Out,
        ZipDivAddOne,
    ): Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out as Container>::Content,
            ZipDivAddOne,
        >,
    //
    (
        <(
            <(
                T,
                <(
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                    U1,
                ) as Sub>::Out,
            ) as TakeFragment>::Out,
            List<(<K as Kernel<D>>::M, Empty)>,
        ) as Mappend>::Out,
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out,
            ZipDivAddOne,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out as Container>::Content,
            ZipDivAddOne,
        >>::Out,
    ): Mappend,
    //
    <(
        <(
            <(
                T,
                <(
                    <(
                        <T as Shape>::Rank,
                        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                    ) as Sub>::Out,
                    U1,
                ) as Sub>::Out,
            ) as TakeFragment>::Out,
            List<(<K as Kernel<D>>::M, Empty)>,
        ) as Mappend>::Out,
        <(
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out,
            ZipDivAddOne,
        ) as Map<
            <<(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out,
                    ZipAdd,
                ) as Map<
                    <<(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out,
                            ZipAdd,
                        ) as Map<
                            <<(
                                <(
                                    <(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out,
                                    ZipSub,
                                ) as Map<
                                    <<(
                                        <(
                                            T,
                                            <(
                                                <T as Shape>::Rank,
                                                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                            ) as Sub>::Out,
                                        ) as SkipFragment>::Out,
                                        <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<
                                            <<K as Kernel<D>>::DilateZipped as Container>::Content,
                                            ZipSubOneMul,
                                        >>::Out,
                                    ) as Zippable>::Out as Container>::Content,
                                    ZipSub,
                                >>::Out,
                                P1,
                            ) as Zippable>::Out as Container>::Content,
                            ZipAdd,
                        >>::Out,
                        P2,
                    ) as Zippable>::Out as Container>::Content,
                    ZipAdd,
                >>::Out,
                S,
            ) as Zippable>::Out as Container>::Content,
            ZipDivAddOne,
        >>::Out,
    ) as Mappend>::Out: ShapeFragment,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        List<(<K as Kernel<D>>::M, Empty)>,
    ): Mappend,
{
    type Out = TensorShape<
        <(
            <(
                <(
                    T,
                    <(
                        <(
                            <T as Shape>::Rank,
                            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                        ) as Sub>::Out,
                        U1,
                    ) as Sub>::Out,
                ) as TakeFragment>::Out,
                List<(<K as Kernel<D>>::M, Empty)>,
            ) as Mappend>::Out,
            <(
                <(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    S,
                ) as Zippable>::Out,
                ZipDivAddOne,
            ) as Map<
                <<(
                    <(
                        <(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out,
                        ZipAdd,
                    ) as Map<
                        <<(
                            <(
                                <(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out,
                                ZipAdd,
                            ) as Map<
                                <<(
                                    <(
                                        <(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out,
                                        ZipSub,
                                    ) as Map<
                                        <<(
                                            <(
                                                T,
                                                <(
                                                    <T as Shape>::Rank,
                                                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                                                ) as Sub>::Out,
                                            ) as SkipFragment>::Out,
                                            <(<K as Kernel<D>>::DilateZipped, ZipSubOneMul) as Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>>::Out,
                                        ) as Zippable>::Out as Container>::Content,
                                        ZipSub,
                                    >>::Out,
                                    P1,
                                ) as Zippable>::Out as Container>::Content,
                                ZipAdd,
                            >>::Out,
                            P2,
                        ) as Zippable>::Out as Container>::Content,
                        ZipAdd,
                    >>::Out,
                    S,
                ) as Zippable>::Out as Container>::Content,
                ZipDivAddOne,
            >>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq, list,
        num::consts::{U1, U2, U3, U4, U5},
    };

    use super::*;
    use crate::shape;
    use typosaurus::bool::True;

    #[allow(unused)]
    #[test]
    fn compat() {
        type T = shape![U1, U2, U5, U5];
        type K = shape![U4, U2, U3, U3];
        type P1 = list![U1, U1];
        type P2 = list![U1, U1];
        type S = list![U1, U1];
        type D = list![U1, U1];

        assert_type_eq!(<(T, K, P1, P2, S, D) as IsCompatible>::Out, True);
        assert_type_eq!(
            shape![U1, U4, U5, U5],
            <(T, K, P1, P2, S, D) as Compatible>::Out
        );

        type ZeroPad = list![U0, U0];
        assert_type_eq!(<(T, K, ZeroPad, ZeroPad, S, D) as IsCompatible>::Out, True);
        assert_type_eq!(
            shape![U1, U4, U3, U3],
            <(T, K, ZeroPad, ZeroPad, S, D) as Compatible>::Out
        );
    }
}



================================================
FILE: src/op/flatten.rs
================================================
use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::{fold::Foldable, semigroup::Mappend},
};

use crate::{
    DecimalDiagnostic, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::{Add, Sub, monoid::Multiplication},
};

struct Flatten;
impl diagnostic::Operation for Flatten {}

/// Boolean type operator for `Flatten` compatibility.
///
/// If shape `T` may be flattened from dimensions `D1` (inclusive) to `D2` (inclusive),
/// then the `Out` associated type of this trait for `(T, D1, D2) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, D1, D2> IsCompatible for (TensorShape<T>, D1, D2)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, D2): IsGreater,
    (D2, D1): IsGreater,
    (
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Flatten` compatible arguments.
///
/// If shape `T` may be flattened from dimensions `D1` (inclusive) to `D2` (inclusive),
/// then the `Out` associated type of this trait for `(T, D1, D2) is the resulting
/// shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, D1, D2> Compatible for (TensorShape<T>, D1, D2)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, D2): IsGreater,
    (D2, D1): IsGreater,
    (
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ): And,
    D1: DecimalDiagnostic,
    <(TensorShape<T>, D1, D2) as IsCompatible>::Out:
        Truthy<Flatten, <TensorShape<T> as ShapeDiagnostic>::Out, <D1 as DecimalDiagnostic>::Out>,
    (TensorShape<T>, D1): Dimensioned,
    (TensorShape<T>, D2): Dimensioned,
    (TensorShape<T>, D1): TakeFragment,
    (TensorShape<T>, D1): SkipFragment,
    (D2, U1): Add,
    (TensorShape<T>, <(D2, U1) as Add>::Out): SkipFragment,
    (<(D2, U1) as Add>::Out, D1): Sub,
    <(TensorShape<T>, D1) as SkipFragment>::Out: ShapeFragment,
    (
        TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
        <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
    ): TakeFragment,
    <(
        TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
        <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
    ) as TakeFragment>::Out: Foldable<Multiplication>,
    (
        <(TensorShape<T>, D1) as TakeFragment>::Out,
        List<(
            <<(
                TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
            ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
            Empty,
        )>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, D1) as TakeFragment>::Out,
            List<(
                <<(
                    TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                    <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
                ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, D1) as TakeFragment>::Out,
            List<(
                <<(
                    TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                    <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
                ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, D1) as TakeFragment>::Out,
                List<(
                    <<(
                        TensorShape<<(TensorShape<T>, D1) as SkipFragment>::Out>,
                        <(<(D2, U1) as Add>::Out, D1) as Sub>::Out,
                    ) as TakeFragment>::Out as Foldable<Multiplication>>::Out,
                    Empty,
                )>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U2, U3, U4, U6, U12, U24},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U4, U2];
        assert_type_eq!(<(MyShape, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![U24]);
        assert_type_eq!(<(MyShape, U0, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U12, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U3, U2, B];

        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U6, B]);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![B]);
        assert_type_eq!(<(MyShape, U1, U2) as Compatible>::Out, shape![U3, B]);
    }
}



================================================
FILE: src/op/gather.rs
================================================
use core::ops::Add;

use typosaurus::{bool::And, num::consts::U1};

use crate::{
    DecimalDiagnostic, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::{IsEqual, IsGreater},
    diagnostic::{self, Truthy},
};

struct Gather;
impl diagnostic::Operation for Gather {}

/// Boolean type operator for `Gather` compatibility.
///
/// If shape `T` may be gathered at dim `I` with indices `U`,
/// then the `Out` associated type of this trait for
/// `(T, I, U)` is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U, I> IsCompatible for (TensorShape<T>, TensorShape<U>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    TensorShape<U>: Shape + ShapeDiagnostic,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank): IsEqual,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<T as ShapeFragment>::Rank, <U as ShapeFragment>::Rank) as IsEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Gather`-compatible shapes.
///
/// If shape `T` may be gathered on dim `I` using indices from shape `U`,
/// then the `Out` associated type of this trait for `(T, I, U)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U, I> Compatible for (TensorShape<T>, TensorShape<U>, I)
where
    (TensorShape<T>, TensorShape<U>, I): IsCompatible,
    <(TensorShape<T>, TensorShape<U>, I) as IsCompatible>::Out: Truthy<Gather, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
{
    type Out = TensorShape<U>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U2, U3, U6, U42, U420},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U42, U2];
        type Another = shape![U6, U6, U2];
        assert_type_eq!(<(MyShape, Another, U1) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, Another, U1) as Compatible>::Out,
            shape![U6, U6, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U2, U42, B, U2];
        type Another = shape![U2, U42, B, U1];
        assert_type_eq!(<(MyShape, Another, U3) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, Another, U2) as Compatible>::Out,
            shape![U2, U42, B, U1]
        );

        type Invalid = shape![U420, U420];
        assert_type_eq!(<(MyShape, Invalid, U1) as IsCompatible>::Out, False);
    }
}



================================================
FILE: src/op/matmul.rs
================================================
use core::ops::Sub;

use typosaurus::bool::And;
use typosaurus::collections::list::{Empty, List as D};
use typosaurus::num::consts::{U1, U2};
use typosaurus::traits::semigroup::Mappend;

use crate::ShapeDiagnostic;
use crate::{
    diagnostic::{self, Truthy},
    Dimension, Dimensioned, IsDimEqual, IsFragEqual, IsRankEqual, Shape, ShapeFragment,
    TakeFragment,
};

pub trait Operand: Sized + Shape {
    type Pre: ShapeFragment;
    type LastDim: Dimension;
    type NextDim: Dimension;
    crate::private!();
}
impl<T> Operand for T
where
    T: Shape,
    <T as Shape>::Rank: Sub<U2> + Sub<U1>,
    (T, <<T as Shape>::Rank as Sub<U2>>::Output): Dimensioned + TakeFragment,
    (T, <<T as Shape>::Rank as Sub<U1>>::Output): Dimensioned,
{
    type Pre = <(T, <<T as Shape>::Rank as Sub<U2>>::Output) as TakeFragment>::Out;
    type LastDim = <(T, <<T as Shape>::Rank as Sub<U1>>::Output) as Dimensioned>::Out;
    type NextDim = <(T, <<T as Shape>::Rank as Sub<U2>>::Output) as Dimensioned>::Out;
    crate::private_impl!();
}

/// Boolean type operator for `Matmul` compatibility.
///
/// If shapes `T` and `U` are compatible with matrix multiplication,
/// then the `Out` associated type of this trait for `(T, U)` is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (T, U)
where
    T: Operand,
    U: Operand,
    (<T as Shape>::Rank, <U as Shape>::Rank): IsRankEqual,
    (<T as Operand>::Pre, <U as Operand>::Pre): IsFragEqual,
    (<T as Operand>::LastDim, <U as Operand>::NextDim): IsDimEqual,
    (
        <(<T as Shape>::Rank, <U as Shape>::Rank) as IsRankEqual>::Out,
        <(<T as Operand>::Pre, <U as Operand>::Pre) as IsFragEqual>::Out,
    ): And,
    (
        <(
            <(<T as Shape>::Rank, <U as Shape>::Rank) as IsRankEqual>::Out,
            <(<T as Operand>::Pre, <U as Operand>::Pre) as IsFragEqual>::Out,
        ) as And>::Out,
        <(<T as Operand>::LastDim, <U as Operand>::NextDim) as IsDimEqual>::Out,
    ): And,
{
    type Out = <(
        <(
            <(<T as Shape>::Rank, <U as Shape>::Rank) as IsRankEqual>::Out,
            <(<T as Operand>::Pre, <U as Operand>::Pre) as IsFragEqual>::Out,
        ) as And>::Out,
        <(<T as Operand>::LastDim, <U as Operand>::NextDim) as IsDimEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Matmul`-compatibile shapes.
///
/// If shapes `T` and `U` may be used together for matrix multiplication,
/// then the `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
struct MatrixMultiplication;
impl diagnostic::Operation for MatrixMultiplication {}

pub trait Compatible {
    type Out: ShapeFragment;
    crate::private!();
}
impl<T, U> Compatible for (T, U)
where
    T: Operand + ShapeDiagnostic,
    U: Operand + ShapeDiagnostic,
    (T, U): IsCompatible,
    <(T, U) as IsCompatible>::Out: Truthy<
        MatrixMultiplication,
        <T as crate::ShapeDiagnostic>::Out,
        <U as crate::ShapeDiagnostic>::Out,
    >,
    (
        D<(<T as Operand>::NextDim, Empty)>,
        D<(<U as Operand>::LastDim, Empty)>,
    ): Mappend,
    (
        <T as Operand>::Pre,
        <(
            D<(<T as Operand>::NextDim, Empty)>,
            D<(<U as Operand>::LastDim, Empty)>,
        ) as Mappend>::Out,
    ): Mappend,
    <(
        <T as Operand>::Pre,
        <(
            D<(<T as Operand>::NextDim, Empty)>,
            D<(<U as Operand>::LastDim, Empty)>,
        ) as Mappend>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = <(
        <T as Operand>::Pre,
        <(
            D<(<T as Operand>::NextDim, Empty)>,
            D<(<U as Operand>::LastDim, Empty)>,
        ) as Mappend>::Out,
    ) as Mappend>::Out;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        num::consts::{U0, U1, U100, U1024, U120, U2, U200, U2048, U240, U3, U360, U3600},
    };

    use super::*;
    use crate::{dynamic::Any, fragment, shape, Dyn};
    use typosaurus::bool::{False, True};
    use typosaurus::collections::list::Idx;

    #[allow(unused)]
    #[test]
    fn compat() {
        type ShapeA = shape![U1, U2, U3];
        type ShapeB = shape![U1, U3, U2];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(Idx<OutShape, U0>, U1);
        assert_type_eq!(Idx<OutShape, U1>, U2);
        assert_type_eq!(Idx<OutShape, U2>, U2);
        assert_type_eq!(OutShape, fragment![U1, U2, U2]);
    }

    #[allow(unused)]
    #[test]
    fn compat2() {
        type ShapeA = shape![U100, U200, U240, U360];
        type ShapeB = shape![U100, U200, U360, U120];
        type PreA = <ShapeA as Operand>::Pre;
        type PreB = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreA, PreB);
        assert_type_eq!(<(PreA, PreB) as IsFragEqual>::Out, True);
        assert_type_eq!(
            <(ShapeA, ShapeB) as Compatible>::Out,
            fragment![U100, U200, U240, U120]
        );
    }

    #[allow(unused)]
    #[test]
    fn compat3() {
        type ShapeA =
            shape![U100, U200, U100, U200, U100, U200, U100, U200, U100, U200, U2048, U3600];
        type ShapeB =
            shape![U100, U200, U100, U200, U100, U200, U100, U200, U100, U200, U3600, U1024];
        type PreA = <ShapeA as Operand>::Pre;
        type PreB = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreA, PreB);
        assert_type_eq!(<(PreA, PreB) as IsFragEqual>::Out, True);
        assert_type_eq!(
            <(ShapeA, ShapeB) as Compatible>::Out,
            fragment![U100, U200, U100, U200, U100, U200, U100, U200, U100, U200, U2048, U1024]
        );
    }

    #[allow(unused)]
    #[test]
    fn compat4() {
        type ShapeA = shape![U2, U3];
        type ShapeB = shape![U3, U2];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(Idx<OutShape, U0>, U2);
        assert_type_eq!(Idx<OutShape, U1>, U2);
        assert_type_eq!(OutShape, fragment![U2, U2]);
    }

    #[allow(unused)]
    #[test]
    fn incompat() {
        type ShapeA = shape![U3, U3, U3, U3, U2, U3];
        type ShapeB = shape![U2, U2, U2, U2, U3, U2];
        type PreL = <ShapeA as Operand>::Pre;
        assert_type_eq!(PreL, fragment![U3, U3, U3, U3]);
        type PreR = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreR, fragment![U2, U2, U2, U2]);
        assert_type_eq!(<(PreL, PreR) as IsFragEqual>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn incompat2() {
        type ShapeA = shape![U3, U3, U2, U3];
        type ShapeB = shape![U2, U2, U3, U3, U3, U2];
        type PreL = <ShapeA as Operand>::Pre;
        assert_type_eq!(PreL, fragment![U3, U3]);
        type PreR = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreR, fragment![U2, U2, U3, U3]);
        assert_type_eq!(<(PreL, PreR) as IsFragEqual>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn incompat3() {
        type ShapeA = shape![U2, U2, U3, U3, U3, U2];
        type ShapeB = shape![U3, U3, U2, U3];
        type PreL = <ShapeA as Operand>::Pre;
        assert_type_eq!(PreL, fragment![U2, U2, U3, U3]);
        type PreR = <ShapeB as Operand>::Pre;
        assert_type_eq!(PreR, fragment![U3, U3]);
        assert_type_eq!(<(PreL, PreR) as IsFragEqual>::Out, False);

        //type Incomp = <(ShapeA, ShapeB) as Compatible>::Out;
        //assert_type_eq!(Incomp, fragment![U3, U3]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type ShapeA = shape![U1, B, U3];
        type ShapeB = shape![U1, U3, B];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(OutShape, fragment![U1, B, B]);
    }

    #[allow(unused)]
    #[test]
    fn wild2() {
        type B = Dyn<Any>;
        type ShapeA = shape![U1, U2, B];
        type ShapeB = shape![U1, U3, U2];

        type OutShape = <(ShapeA, ShapeB) as Compatible>::Out;
        assert_type_eq!(OutShape, fragment![U1, U2, U2]);
    }
}



================================================
FILE: src/op/mod.rs
================================================
pub mod broadcast;
pub mod cat;
pub mod cat_dyn;
pub mod convolution;
pub mod flatten;
pub mod gather;
pub mod matmul;
pub mod narrow;
pub mod narrow_dyn;
pub mod narrow_dyn_start;
pub mod pad;
pub mod permute;
pub mod reshape;
pub mod squeeze;
pub mod stack;
pub mod transpose;
pub mod unsqueeze;



================================================
FILE: src/op/narrow.rs
================================================
use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    cmp::{IsGreater, IsGreaterOrEqual},
    diagnostic::{self, Truthy},
    num::Add,
    DecimalDiagnostic, Dimension, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, Tensor, TensorShape, IDX,
};

pub struct Narrow;
impl diagnostic::Operation for Narrow {}

pub fn check<T, TS, I, S, L>(_t: &T)
where
    T: Tensor<Shape = TS>,
    TS: ShapeDiagnostic,
    I: DecimalDiagnostic,
    (TS, I, S, L): IsCompatible,
    <(TS, I, S, L) as IsCompatible>::Out:
        Truthy<Narrow, <TS as ShapeDiagnostic>::Out, <I as DecimalDiagnostic>::Out>,
{
}

/// Boolean type operator for `Narrow` compatibility.
///
/// If shape `T` may be narrowed at dim `I` to length `L` starting
/// from element `S`, then the `Out` associated type of this trait for
/// `(T, I, S, L) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I, L, S> IsCompatible for (TensorShape<T>, I, S, L)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (S, L): Add,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(S, L) as Add>::Out,
    ): IsGreaterOrEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(
            <(TensorShape<T>, I) as Dimensioned>::Out,
            <(S, L) as Add>::Out,
        ) as IsGreaterOrEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(
            <(TensorShape<T>, I) as Dimensioned>::Out,
            <(S, L) as Add>::Out,
        ) as IsGreaterOrEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Narrow`-compatible shapes.
///
/// If shape `T` may be narrowed on dim `I` to length `L` starting from element
/// `S`, then the `Out` associated type of this trait for `(T, I, S, L)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, S, L> Compatible for (TensorShape<T>, I, S, L)
where
    (TensorShape<T>, I, S, L): IsCompatible,
    <(TensorShape<T>, I, S, L) as IsCompatible>::Out: Truthy<
        Narrow,
        <TensorShape<T> as ShapeDiagnostic>::Out,
        IDX<<I as DecimalDiagnostic>::Out>,
    >,
    I: DecimalDiagnostic,
    L: Dimension,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (S, L): Add,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(S, L) as Add>::Out,
    ): IsGreaterOrEqual,
    (<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>): Mappend,
    (
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U2, U3, U42, U6},
    };

    use super::*;

    use crate::{dynamic::Any, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U0, U1, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U0, U1, U2) as Compatible>::Out,
            shape![U2, U1, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U42, B, U1];
        assert_type_eq!(<(MyShape, U1, U6, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, U6, U6) as Compatible>::Out,
            shape![U1, U6, B, U1]
        );

        assert_type_eq!(<(MyShape, U2, U6, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, U6, U2) as Compatible>::Out,
            shape![U1, U42, U2, U1]
        );
    }
}



================================================
FILE: src/op/narrow_dyn.rs
================================================
use core::ops::Add;

use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
};

struct Narrow;
impl diagnostic::Operation for Narrow {}

/// Boolean type operator for `Narrow` compatibility.
///
/// If shape `T` may be narrowed at dim `I` to length `L` starting
/// from element 0, then the `Out` associated type of this trait for
/// `(T, I, L) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I, DynDim> IsCompatible for (TensorShape<T>, I, DynDim)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out;
    crate::private_impl!();
}

/// Type operator for `Narrow`-compatible shapes.
///
/// If shape `T` may be narrowed on dim `I` to length `L` starting from element
/// 0, then the `Out` associated type of this trait for `(T, I, L)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, DynDim> Compatible for (TensorShape<T>, I, DynDim)
where
    (TensorShape<T>, I, DynDim): IsCompatible,
    <(TensorShape<T>, I, DynDim) as IsCompatible>::Out: Truthy<Narrow, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(DynDim, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(DynDim, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(DynDim, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(DynDim, Empty)>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U2, U3},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, dyndims, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        type D = Dyn<Any>;
        assert_type_eq!(<(MyShape, U0, Dyn<Any>) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U0, Dyn<Any>) as Compatible>::Out,
            shape![D, U1, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn dynamic() {
        dyndims! {
            N: SequenceLength,
            B: BatchSize
        }
        type DynShape = shape![B, N, U3];
        assert_type_eq!(<(DynShape, U1, N) as IsCompatible>::Out, True);
    }
}



================================================
FILE: src/op/narrow_dyn_start.rs
================================================
use core::ops::Add;

use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimension, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment,
    SkipFragment, TakeFragment, TensorShape,
    cmp::{IsGreater, IsGreaterOrEqual},
    diagnostic::{self, Truthy},
};

struct Narrow;
impl diagnostic::Operation for Narrow {}

/// Boolean type operator for `Narrow` compatibility.
///
/// If shape `T` may be narrowed at dim `I` to length `L` starting
/// from element 0, then the `Out` associated type of this trait for
/// `(T, I, L) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I, L> IsCompatible for (TensorShape<T>, I, L)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as Dimensioned>::Out, L): IsGreaterOrEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, L) as IsGreaterOrEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, L) as IsGreaterOrEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Narrow`-compatible shapes.
///
/// If shape `T` may be narrowed on dim `I` to length `L` starting from element
/// 0, then the `Out` associated type of this trait for `(T, I, L)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, L> Compatible for (TensorShape<T>, I, L)
where
    (TensorShape<T>, I, L): IsCompatible,
    <(TensorShape<T>, I, L) as IsCompatible>::Out: Truthy<Narrow, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    L: Dimension,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as Dimensioned>::Out, L): IsGreaterOrEqual,
    (<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>): Mappend,
    (
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(<(TensorShape<T>, I) as TakeFragment>::Out, List<(L, Empty)>) as Mappend>::Out,
            <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::True,
        num::consts::{U0, U1, U2, U3, U6, U42},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![U2, U1, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U42, B, U1];
        assert_type_eq!(<(MyShape, U1, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, U6) as Compatible>::Out,
            shape![U1, U6, B, U1]
        );

        assert_type_eq!(<(MyShape, U2, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, U2) as Compatible>::Out,
            shape![U1, U42, U2, U1]
        );
    }
}



================================================
FILE: src/op/pad.rs
================================================
use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimension, Dimensioned, IDX, Shape, ShapeDiagnostic, ShapeFragment,
    SkipFragment, TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::Add,
};

struct Pad;
impl diagnostic::Operation for Pad {}

/// Boolean type operator for `Narrow` compatibility.
///
/// If shape `T` may be narrowed at dim `I` to length `L` starting
/// from element `S`, then the `Out` associated type of this trait for
/// `(T, I, S, L) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I, A, B> IsCompatible for (TensorShape<T>, I, A, B)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (A, B): Add,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out;
    crate::private_impl!();
}

/// Type operator for `Narrow`-compatible shapes.
///
/// If shape `T` may be narrowed on dim `I` to length `L` starting from element
/// `S`, then the `Out` associated type of this trait for `(T, I, S, L)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I, A, B> Compatible for (TensorShape<T>, I, A, B)
where
    (TensorShape<T>, I, A, B): IsCompatible,
    <(TensorShape<T>, I, A, B) as IsCompatible>::Out:
        Truthy<Pad, <TensorShape<T> as ShapeDiagnostic>::Out, IDX<<I as DecimalDiagnostic>::Out>>,
    I: DecimalDiagnostic,
    A: Dimension,
    B: Dimension,
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    (I, U1): Add,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <(I, U1) as Add>::Out): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (A, B): Add,
    (
        <(TensorShape<T>, I) as Dimensioned>::Out,
        <(A, B) as Add>::Out,
    ): Add,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(
            <(
                <(TensorShape<T>, I) as Dimensioned>::Out,
                <(A, B) as Add>::Out,
            ) as Add>::Out,
            Empty,
        )>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(
                <(
                    <(TensorShape<T>, I) as Dimensioned>::Out,
                    <(A, B) as Add>::Out,
                ) as Add>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(
                <(
                    <(TensorShape<T>, I) as Dimensioned>::Out,
                    <(A, B) as Add>::Out,
                ) as Add>::Out,
                Empty,
            )>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(
                    <(
                        <(TensorShape<T>, I) as Dimensioned>::Out,
                        <(A, B) as Add>::Out,
                    ) as Add>::Out,
                    Empty,
                )>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(I, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U0, U1, U2, U4, U6, U8},
    };

    use super::*;

    use crate::shape;

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U2, U4, U6, U8];
        assert_type_eq!(<(MyShape, U1, U2, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U1, U2, U2) as Compatible>::Out,
            shape![U2, U8, U6, U8]
        );
        assert_type_eq!(<(MyShape, U0, U4, U0) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U0, U4, U0) as Compatible>::Out,
            shape![U6, U4, U6, U8]
        );
        assert_type_eq!(<(MyShape, U2, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, U2, U0, U2) as Compatible>::Out,
            shape![U2, U4, U8, U8]
        );

        assert_type_eq!(<(MyShape, U6, U2, U2) as IsCompatible>::Out, False);
        assert_type_eq!(<(MyShape, U8, U0, U0) as IsCompatible>::Out, False);
    }
}



================================================
FILE: src/op/permute.rs
================================================
use typosaurus::{
    bool::{And, monoid::Both},
    collections::{
        Container,
        list::{IsUnique, Len, Ones},
    },
    num::Addition,
    traits::{fold::Foldable, functor::Map},
};

use crate::{
    AllLessThan, IsLessThan, PermutationOf, Shape, ShapeDiagnostic, ShapeFragment,
    cmp::IsEqual,
    diagnostic::{self, Truthy},
};

/// Boolean type operator for `Permute` compatibility.
///
/// If shape `T` may be permuted using dimension indices represented by `U`,
/// then the `Out` associated type of this trait for `(T, U)` is `True`.
/// Otherwise, it is `False`.
struct Permute;
impl diagnostic::Operation for Permute {}

pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (T, U)
where
    (U, Ones): Map<<U as Container>::Content, Ones>,
    <(U, Ones) as Map<<U as Container>::Content, Ones>>::Out: Foldable<Addition>,
    T: Shape + ShapeDiagnostic,
    (<T as Shape>::Rank, Len<U>): IsEqual,
    U: IsUnique,
    U: Container,
    (U, IsLessThan<<T as Shape>::Rank>):
        Map<<U as Container>::Content, IsLessThan<<T as Shape>::Rank>>,
    <(U, IsLessThan<<T as Shape>::Rank>) as Map<
        <U as Container>::Content,
        IsLessThan<<T as Shape>::Rank>,
    >>::Out: Foldable<Both>,
    (
        <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out,
        <U as IsUnique>::Out,
    ): And,
    (
        <(
            <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out,
            <U as IsUnique>::Out,
        ) as And>::Out,
        AllLessThan<U, <T as Shape>::Rank>,
    ): And,
{
    type Out = <(
        <(
            <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out,
            <U as IsUnique>::Out,
        ) as And>::Out,
        AllLessThan<U, <T as Shape>::Rank>,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Permute`-compatible shapes.
///
/// If shape `T` may be permuted by dimension indices `U`, then the
/// `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
pub trait Compatible {
    type Out: ShapeFragment;
    crate::private!();
}
impl<T, U> Compatible for (T, U)
where
    (U, Ones): Map<<U as Container>::Content, Ones>,
    <(U, Ones) as Map<<U as Container>::Content, Ones>>::Out: Foldable<Addition>,
    T: Shape + ShapeDiagnostic,
    (<T as Shape>::Rank, Len<U>): IsEqual,
    <(<T as Shape>::Rank, Len<U>) as IsEqual>::Out:
        Truthy<Permute, <T as crate::ShapeDiagnostic>::Out, ()>,
    U: IsUnique,
    <U as IsUnique>::Out: Truthy<Permute, <T as crate::ShapeDiagnostic>::Out, ()>,
    U: Container,
    (U, IsLessThan<<T as Shape>::Rank>):
        Map<<U as Container>::Content, IsLessThan<<T as Shape>::Rank>>,
    <(U, IsLessThan<<T as Shape>::Rank>) as Map<
        <U as Container>::Content,
        IsLessThan<<T as Shape>::Rank>,
    >>::Out: Foldable<Both>,
    AllLessThan<U, <T as Shape>::Rank>: Truthy<Permute, <T as crate::ShapeDiagnostic>::Out, ()>,
    (U, PermutationOf<T>): Map<<U as Container>::Content, PermutationOf<T>>,
    <(U, PermutationOf<T>) as Map<<U as Container>::Content, PermutationOf<T>>>::Out: ShapeFragment,
{
    type Out = <(U, PermutationOf<T>) as Map<<U as Container>::Content, PermutationOf<T>>>::Out;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        list,
        num::consts::{U0, U1, U2, U3, U4, U5, U42},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, fragment, shape};

    #[allow(unused)]
    #[test]
    fn permute() {
        type T = shape![U1, U1, U42];
        type I = list![U2, U0, U1];
        assert_type_eq!(<(T, I) as IsCompatible>::Out, True);
        assert_type_eq!(<(T, I) as Compatible>::Out, fragment![U42, U1, U1]);
    }

    #[allow(unused)]
    #[test]
    fn permute2() {
        type T = shape![U1, U2, U42, U1, U42, U2];
        type I1 = list![U2, U0, U1];
        assert_type_eq!(<(T, I1) as IsCompatible>::Out, False);

        type I2 = list![U5, U4, U3, U2, U1, U0];
        assert_type_eq!(<(T, I2) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(T, I2) as Compatible>::Out,
            fragment![U2, U42, U1, U42, U2, U1]
        );
    }

    #[allow(unused)]
    #[test]
    fn permute3() {
        type MyShape = shape![U2, U3, U4, U5];
        type Indices = list![U2, U3, U1, U0];
        assert_type_eq!(
            <(MyShape, Indices) as Compatible>::Out,
            fragment![U4, U5, U3, U2]
        );

        type Another = shape![U2, U3];
        type Idxs = list![U1, U0];
        assert_type_eq!(<(Another, Idxs) as Compatible>::Out, fragment![U3, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B, U1];
        type Indices = list![U2, U3, U1, U0];
        assert_type_eq!(<(MyShape, Indices) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, Indices) as Compatible>::Out,
            fragment![B, U1, U1, U1]
        );
    }
}



================================================
FILE: src/op/reshape.rs
================================================
use typosaurus::{collections::Container, traits::fold::Foldable};

use crate::{
    cmp::IsEqual,
    diagnostic::{self, Truthy},
    num::monoid::Multiplication,
    Product, Shape, ShapeDiagnostic, ShapeFragment, Tensor, TensorShape,
};

pub struct Reshape;
impl diagnostic::Operation for Reshape {}

pub fn check<T1, S1, S2>(_t1: &T1)
where
    T1: Tensor<Shape = S1>,
    S1: ShapeDiagnostic,
    S2: ShapeDiagnostic,
    (S2, S1): IsCompatible,
    <(S2, S1) as IsCompatible>::Out:
        Truthy<Reshape, <S1 as ShapeDiagnostic>::Out, <S2 as ShapeDiagnostic>::Out>,
{
}

/// Boolean type operator for `Reshape` compatibility.
///
/// If shape `T` may be reshaped to shape `U`, then the `Out`
/// associated type of this trait for `(T, U) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment + Container,
    U: ShapeFragment + Container,
    T: Foldable<Multiplication>,
    U: Foldable<Multiplication>,
    (Product<T>, Product<U>): IsEqual,
{
    type Out = <(Product<T>, Product<U>) as IsEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Reshape`-compatible shapes.
///
/// If shape `T` may be reshaped to shape `U`, then the
/// `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U> Compatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment + Container,
    U: ShapeFragment + Container,
    T: Foldable<Multiplication>,
    U: Foldable<Multiplication>,
    (Product<T>, Product<U>): IsEqual,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out:
        Truthy<Reshape, TensorShape<T>, TensorShape<U>>,
{
    type Out = TensorShape<U>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U1, U2, U3, U6, U7},
    };

    use super::*;

    use crate::{dynamic::Any, dyndims, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U1, U6]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U1, U6]) as Compatible>::Out,
            shape![U1, U6]
        );

        assert_type_eq!(<(MyShape, shape![U2, U3]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U2, U3]) as Compatible>::Out,
            shape![U2, U3]
        );

        assert_type_eq!(<(MyShape, shape![U2, U3, U1]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U2, U3, U1]) as Compatible>::Out,
            shape![U2, U3, U1]
        );
    }

    #[allow(unused)]
    #[test]
    fn invalid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U1, U7]) as IsCompatible>::Out, False);
        assert_type_eq!(<(MyShape, shape![U3, U3]) as IsCompatible>::Out, False);
        assert_type_eq!(<(MyShape, shape![U2, U3, U2]) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B, U1];
        assert_type_eq!(<(MyShape, shape![U1, U7]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U1, U2, U3, U2, U7]) as IsCompatible>::Out,
            True
        );

        assert_type_eq!(<(MyShape, shape![U1]) as IsCompatible>::Out, True);
    }

    #[allow(unused)]
    #[test]
    fn dynamic() {
        dyndims! {
            B: BatchSize,
            N: SequenceLength
        }

        type MyShape = shape![B, N, U2, U3];
        assert_type_eq!(<(MyShape, shape![B, N, U6]) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, shape![U1, U6, N, B]) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, shape![B, N, U7]) as IsCompatible>::Out, False);
        assert_type_eq!(
            <(MyShape, shape![U1, U7, N, B]) as IsCompatible>::Out,
            False
        );
    }
}



================================================
FILE: src/op/squeeze.rs
================================================
use core::ops::Add;

use typosaurus::num::consts::U1;
use typosaurus::{bool::And, traits::semigroup::Mappend};

use crate::Tensor;
use crate::{
    cmp::{IsEqual, IsGreater},
    diagnostic::{self, Truthy},
    DecimalDiagnostic, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
};

pub struct Squeeze;
impl diagnostic::Operation for Squeeze {}

pub fn check<T1, S1, I>(_t1: &T1)
where
    T1: Tensor<Shape = S1>,
    S1: ShapeDiagnostic,
    I: DecimalDiagnostic,
    (S1, I): IsCompatible,
    <(S1, I) as IsCompatible>::Out:
        Truthy<Squeeze, <S1 as ShapeDiagnostic>::Out, <I as DecimalDiagnostic>::Out>,
{
}

/// Boolean type operator for `Squeeze` compatibility.
///
/// If shape `T` may be squeezed at dim `I`,
/// then the `Out` associated type of this trait for
/// `(T, I) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I> IsCompatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreater,
    I: Add<U1>,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): Dimensioned,
    (<(TensorShape<T>, I) as Dimensioned>::Out, U1): IsEqual,
    (
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, U1) as IsEqual>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, I) as IsGreater>::Out,
        <(<(TensorShape<T>, I) as Dimensioned>::Out, U1) as IsEqual>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Squeeze`-compatible shapes.
///
/// If shape `T` may be squeezed at dim `I`, then the
/// `Out` associated type of this trait for `(T, I)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I> Compatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    I: Add<U1> + DecimalDiagnostic,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, <I as Add<U1>>::Output): SkipFragment,
    (TensorShape<T>, I): IsCompatible,
    <(TensorShape<T>, I) as IsCompatible>::Out: Truthy<
        Squeeze,
        <TensorShape<T> as crate::ShapeDiagnostic>::Out,
        crate::IDX<<I as crate::DecimalDiagnostic>::Out>,
    >,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(TensorShape<T>, I) as TakeFragment>::Out,
        <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            <(TensorShape<T>, <I as Add<U1>>::Output) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U0, U1, U1000, U151, U2, U3, U4, U5, U6, U64, U936},
    };

    use super::*;

    use crate::{dynamic::Any, shape, Dyn};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as Compatible>::Out, shape![U3, U2]);

        type Another = shape![U1, U2, U3, U2, U3, U2, U1];
        assert_type_eq!(<(Another, U0) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U0) as Compatible>::Out,
            shape![U2, U3, U2, U3, U2, U1]
        );
        assert_type_eq!(<(Another, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U6) as Compatible>::Out,
            shape![U1, U2, U3, U2, U3, U2]
        );
        assert_type_eq!(<(Another, U1) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U2) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U3) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U4) as IsCompatible>::Out, False);
        assert_type_eq!(<(Another, U5) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn logits() {
        type U151936 = <<U151 as core::ops::Mul<U1000>>::Output as Add<U936>>::Output;
        pub type Logits = shape![U1, U1, U151936];
        assert_type_eq!(<(Logits, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(Logits, U0) as Compatible>::Out, shape![U1, U151936]);

        type Test = shape![U1, U64, U1];
        assert_type_eq!(<(Test, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(Test, U0) as Compatible>::Out, shape![U64, U1]);
        assert_type_eq!(<(Test, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(Test, U2) as Compatible>::Out, shape![U1, U64]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B, U1];
        assert_type_eq!(<(MyShape, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0) as Compatible>::Out, shape![U1, B, U1]);
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as Compatible>::Out, shape![U1, B, U1]);
        assert_type_eq!(<(MyShape, U3) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U3) as Compatible>::Out, shape![U1, U1, B]);
    }
}



================================================
FILE: src/op/stack.rs
================================================
use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U2,
    traits::semigroup::Mappend,
};

use crate::{
    IsFragEqual, Shape, ShapeDiagnostic, ShapeFragment, TensorShape,
    diagnostic::{self, Truthy},
    fragment,
};

struct Stack;
impl diagnostic::Operation for Stack {}

/// Boolean type operator for `Stack` compatibility.
///
/// If shapes `T` and `U` may be stacked,
/// then the `Out` associated type of this trait for
/// `(T, U) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, U> IsCompatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): IsFragEqual,
{
    type Out = <(T, U) as IsFragEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Stack`-compatible shapes.
///
/// If shapes `T` and `U` may be stacked, then the
/// `Out` associated type of this trait for `(T, U)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, U> Compatible for (TensorShape<T>, TensorShape<U>)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    TensorShape<U>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    U: ShapeFragment,
    (T, U): IsFragEqual,
    (List<(U2, Empty)>, T): Mappend,
    <(List<(U2, Empty)>, T) as Mappend>::Out: ShapeFragment,
    (TensorShape<T>, TensorShape<U>): IsCompatible,
    <(TensorShape<T>, TensorShape<U>) as IsCompatible>::Out:
        Truthy<Stack, TensorShape<T>, TensorShape<U>>,
{
    type Out = TensorShape<<(fragment![U2], T) as Mappend>::Out>;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::bool::{False, True};
    use typosaurus::{
        assert_type_eq,
        num::consts::{U1, U2, U3, U7},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U3, U2]) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, shape![U3, U2]) as Compatible>::Out,
            shape![U2, U3, U2]
        );
    }

    #[allow(unused)]
    #[test]
    fn invalid() {
        type MyShape = shape![U3, U2];
        assert_type_eq!(<(MyShape, shape![U1, U7]) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B];
        assert_type_eq!(<(MyShape, MyShape) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(MyShape, MyShape) as Compatible>::Out,
            shape![U2, U1, U1, B]
        );
    }
}



================================================
FILE: src/op/transpose.rs
================================================
use typosaurus::{
    bool::And,
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Dimensioned, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment,
    TakeFragment, TensorShape,
    cmp::IsGreater,
    diagnostic::{self, Truthy},
    num::{Add, Sub},
};

struct Transpose;
impl diagnostic::Operation for Transpose {}

/// Boolean type operator for `Transpose` compatibility.
///
/// If shape `T` may be transposed at dims `D1` and `D2`,
/// then the `Out` associated type of this trait for
/// `(T, D1, D2) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, D1, D2> IsCompatible for (TensorShape<T>, D1, D2)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, D2): IsGreater,
    (D2, D1): IsGreater,
    (
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ): And,
{
    type Out = <(
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ) as And>::Out;
    crate::private_impl!();
}

/// Type operator for `Transpose` compatible shapes.
///
/// If dimensions `D1` and `D2` of shape `T` may be transposed, then the
/// `Out` associated type of this trait for `(T, D1, D2)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, D1, D2> Compatible for (TensorShape<T>, D1, D2)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, D2): IsGreater,
    (D2, D1): IsGreater,
    (
        <(<T as ShapeFragment>::Rank, D2) as IsGreater>::Out,
        <(D2, D1) as IsGreater>::Out,
    ): And,
    D1: DecimalDiagnostic,
    <(TensorShape<T>, D1, D2) as IsCompatible>::Out:
        Truthy<Transpose, <TensorShape<T> as ShapeDiagnostic>::Out, <D1 as DecimalDiagnostic>::Out>,
    (TensorShape<T>, D1): Dimensioned,
    (TensorShape<T>, D2): Dimensioned,
    (D1, U1): Add,
    (TensorShape<T>, D1): TakeFragment,
    (TensorShape<T>, <(D1, U1) as Add>::Out): SkipFragment,
    (D2, U1): Add,
    (TensorShape<T>, <(D2, U1) as Add>::Out): SkipFragment,
    (D2, <(D1, U1) as Add>::Out): Sub,
    <(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out: ShapeFragment,
    (
        TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
        <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
    ): TakeFragment,
    (
        <(TensorShape<T>, D1) as TakeFragment>::Out,
        List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, D1) as TakeFragment>::Out,
            List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
        ) as Mappend>::Out,
        <(
            TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
            <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
        ) as TakeFragment>::Out,
    ): Mappend,
    (
        <(
            <(
                <(TensorShape<T>, D1) as TakeFragment>::Out,
                List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
            ) as Mappend>::Out,
            <(
                TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
                <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
            ) as TakeFragment>::Out,
        ) as Mappend>::Out,
        List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
    ): Mappend,
    (
        <(
            <(
                <(
                    <(TensorShape<T>, D1) as TakeFragment>::Out,
                    List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
                ) as Mappend>::Out,
                <(
                    TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
                    <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
                ) as TakeFragment>::Out,
            ) as Mappend>::Out,
            List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(
                <(
                    <(TensorShape<T>, D1) as TakeFragment>::Out,
                    List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
                ) as Mappend>::Out,
                <(
                    TensorShape<<(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out>,
                    <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
                ) as TakeFragment>::Out,
            ) as Mappend>::Out,
            List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(
                    <(
                        <(TensorShape<T>, D1) as TakeFragment>::Out,
                        List<(<(TensorShape<T>, D2) as Dimensioned>::Out, Empty)>,
                    ) as Mappend>::Out,
                    <(
                        TensorShape<
                            <(TensorShape<T>, <(D1, U1) as Add>::Out) as SkipFragment>::Out,
                        >,
                        <(D2, <(D1, U1) as Add>::Out) as Sub>::Out,
                    ) as TakeFragment>::Out,
                ) as Mappend>::Out,
                List<(<(TensorShape<T>, D1) as Dimensioned>::Out, Empty)>,
            ) as Mappend>::Out,
            <(TensorShape<T>, <(D2, U1) as Add>::Out) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::bool::True;
    use typosaurus::{
        assert_type_eq,
        num::consts::{U0, U1, U2, U3},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn valid() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U0, U2) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![U2, U1, U3]);
        assert_type_eq!(<(MyShape, U0, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U1, U3, U2]);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U2, B];

        assert_type_eq!(<(MyShape, U0, U1) as Compatible>::Out, shape![U2, U1, B]);
        assert_type_eq!(<(MyShape, U0, U2) as Compatible>::Out, shape![B, U2, U1]);
        assert_type_eq!(<(MyShape, U1, U2) as Compatible>::Out, shape![U1, B, U2]);
    }
}



================================================
FILE: src/op/unsqueeze.rs
================================================
use typosaurus::{
    collections::list::{Empty, List},
    num::consts::U1,
    traits::semigroup::Mappend,
};

use crate::{
    DecimalDiagnostic, Shape, ShapeDiagnostic, ShapeFragment, SkipFragment, TakeFragment,
    TensorShape,
    cmp::IsGreaterOrEqual,
    diagnostic::{self, Truthy},
};

struct Unsqueeze;
impl diagnostic::Operation for Unsqueeze {}

/// Boolean type operator for `Unsqueeze` compatibility.
///
/// If shape `T` may be unsqueezed at dim `I`,
/// then the `Out` associated type of this trait for
/// `(T, I) is `True`.
/// Otherwise, it is `False`.
pub trait IsCompatible {
    type Out;
    crate::private!();
}
impl<T, I> IsCompatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreaterOrEqual,
{
    type Out = <(<T as ShapeFragment>::Rank, I) as IsGreaterOrEqual>::Out;
    crate::private_impl!();
}

/// Type operator for `Unsqueeze`-compatible shapes.
///
/// If shape `T` may be unsqueezed at dim `I`, then the
/// `Out` associated type of this trait for `(T, I)` is the
/// resulting shape.
pub trait Compatible {
    type Out: Shape;
    crate::private!();
}
impl<T, I> Compatible for (TensorShape<T>, I)
where
    TensorShape<T>: Shape + ShapeDiagnostic,
    T: ShapeFragment,
    (<T as ShapeFragment>::Rank, I): IsGreaterOrEqual,
    (TensorShape<T>, I): TakeFragment,
    (TensorShape<T>, I): SkipFragment,
    (TensorShape<T>, I): IsCompatible,
    I: DecimalDiagnostic,
    <(TensorShape<T>, I) as IsCompatible>::Out: Truthy<
            Unsqueeze,
            <TensorShape<T> as crate::ShapeDiagnostic>::Out,
            crate::IDX<<I as crate::DecimalDiagnostic>::Out>,
        >,
    (
        <(TensorShape<T>, I) as TakeFragment>::Out,
        List<(U1, Empty)>,
    ): Mappend,
    (
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(U1, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, I) as SkipFragment>::Out,
    ): Mappend,
    <(
        <(
            <(TensorShape<T>, I) as TakeFragment>::Out,
            List<(U1, Empty)>,
        ) as Mappend>::Out,
        <(TensorShape<T>, I) as SkipFragment>::Out,
    ) as Mappend>::Out: ShapeFragment,
{
    type Out = TensorShape<
        <(
            <(
                <(TensorShape<T>, I) as TakeFragment>::Out,
                List<(U1, Empty)>,
            ) as Mappend>::Out,
            <(TensorShape<T>, I) as SkipFragment>::Out,
        ) as Mappend>::Out,
    >;
    crate::private_impl!();
}

#[cfg(test)]
mod test {
    use typosaurus::{
        assert_type_eq,
        bool::{False, True},
        num::consts::{U0, U1, U2, U3, U4, U6, U7, U8},
    };

    use super::*;

    use crate::{Dyn, dynamic::Any, shape};

    #[allow(unused)]
    #[test]
    fn basic() {
        type MyShape = shape![U3, U1, U2];
        assert_type_eq!(<(MyShape, U1) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U1) as Compatible>::Out, shape![U3, U1, U1, U2]);

        type Another = shape![U1, U2, U3, U2, U3, U2, U1];
        assert_type_eq!(<(Another, U0) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U0) as Compatible>::Out,
            shape![U1, U1, U2, U3, U2, U3, U2, U1]
        );
        assert_type_eq!(<(Another, U4) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U4) as Compatible>::Out,
            shape![U1, U2, U3, U2, U1, U3, U2, U1]
        );
        assert_type_eq!(<(Another, U6) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U6) as Compatible>::Out,
            shape![U1, U2, U3, U2, U3, U2, U1, U1]
        );
        assert_type_eq!(<(Another, U7) as IsCompatible>::Out, True);
        assert_type_eq!(
            <(Another, U6) as Compatible>::Out,
            shape![U1, U2, U3, U2, U3, U2, U1, U1]
        );
        assert_type_eq!(<(Another, U8) as IsCompatible>::Out, False);
    }

    #[allow(unused)]
    #[test]
    fn wild() {
        type B = Dyn<Any>;
        type MyShape = shape![U1, U1, B];
        assert_type_eq!(<(MyShape, U0) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U0) as Compatible>::Out, shape![U1, U1, U1, B]);

        assert_type_eq!(<(MyShape, U3) as IsCompatible>::Out, True);
        assert_type_eq!(<(MyShape, U3) as Compatible>::Out, shape![U1, U1, B, U1]);
    }
}



================================================
FILE: .github/workflows/ci.yml
================================================
name: Cargo Check & Test

on:
  push:
  pull_request:

env: 
  CARGO_TERM_COLOR: always

jobs:
  check:
    name: Rust project - latest
    runs-on: ubuntu-latest
    strategy:
      matrix:
        toolchain:
          - stable
    steps:
      - uses: actions/checkout@v4
      - run: cargo check --workspace
      - run: cargo test --workspace


