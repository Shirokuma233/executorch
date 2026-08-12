/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run Llama2 110M, Llama3.2 1B / 3B, Gemma 2B, Gemma2 2B, Gemma3
 * 1B, Granite3.3 2B, phi4-mini-instruct, Qwen2.5 0.5B / 1.5B, Qwen3 0.6B
 * / 1.7B, SmolLM2 135M, SmolLM3 3B with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <fstream>
#include <vector>

DEFINE_string(decoder_model_version, "llama2", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model is serialized using the flatbuffer format. If specified, seq_len can exceed the context length defined in the model.");
DEFINE_string(
    output_path,
    "outputs.txt",
    "Executorch inference data output path.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Records inference speed. For CI purpose.");
DEFINE_string(
    dump_logits_path,
    "",
    "If path is provided, program will dump all logits generated. This option is for analysis purpose. It is not recommended for general usage as it will cause token rate drop and increase in memory usage.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");
DEFINE_string(
    prompt,
    "The answer to the ultimate question is",
    "User prompts for Llama. When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.");
DEFINE_string(
    tokenized_prompt,
    "",
    "This is an alternative of passing prompts. Users could provide this in a raw file, with tokens saved in uint64 format.");
DEFINE_string(
    system_prompt,
    "",
    "Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding / 3: Eagle Decoding");
DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy use case between the application and device/co-processor associated with the backend.");
DEFINE_bool(
    enable_thinking,
    false,
    "Qwen3 only: let the model emit its own <think> block. Off by default, matching llama.py, which pre-fills an empty one to skip reasoning.");
DEFINE_int32(num_iters, 1, "total num of iterations to run.");
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] Represents the size of the n-grams used in the lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Determines how many future tokens the algorithm attempts to predict in each step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.");
DEFINE_string(
    eagle_head_path,
    "",
    "[Eagle Decoding] Path to the compiled EAGLE draft head pte.");
DEFINE_int32(
    max_tree_size,
    0,
    "[Eagle Decoding] Max tree nodes including the root (must fit target compiled ar_len).");
DEFINE_int32(
    draft_len,
    0,
    "[Eagle Decoding] Chain-mode draft length (Phase 3).");
DEFINE_int32(
    tree_depth,
    4,
    "[Eagle Decoding] EAGLE tree expansion depth. Set 0 to use chain mode.");
DEFINE_int32(
    tree_topk,
    4,
    "[Eagle Decoding] EAGLE top-k branches retained at each tree expansion.");
DEFINE_string(
    eagle_d2t_path,
    "",
    "[Eagle Decoding] Path to d2t.bin (int64[draft_vocab_size]). "
    "Default: <eagle_head dir>/d2t.bin");
DEFINE_string(
    eagle_t2d_path,
    "",
    "[Eagle Decoding] Path to t2d.bin (bool[target_vocab_size]). "
    "Default: <eagle_head dir>/t2d.bin");
DEFINE_string(
    eagle_embed_path,
    "",
    "[Eagle Decoding] Path to embed.bin (fp16[target_vocab, hidden]). "
    "Default: <eagle_head dir>/embed.bin");
DEFINE_string(
    dflash_draft_path,
    "",
    "[DFlash Decoding] Path to the compiled DFlash draft pte.");
DEFINE_int32(
    block_size,
    16,
    "[DFlash Decoding] Draft block size (drafts block_size-1 tokens per step).");
DEFINE_int32(
    dflash_max_context_len,
    0,
    "[DFlash Decoding] Fixed context length the draft attends to (0 => model).");
DEFINE_int32(
    dflash_tree_budget,
    0,
    "[DFlash Decoding] Non-root nodes in the draft tree (DDTree). 0 keeps the "
    "chain: one argmax per block position, first miss ends the round. Capped at "
    "target_ar_len - 1, since the verify window is compiled, not configurable.");
DEFINE_double(
    dflash_logit_out_scale,
    0.0,
    "[DFlash Decoding] Override the lm_head OUTPUT logit encoding scale. 0 (the "
    "default) takes it from the pte's get_logits_out_scale. Only the tree reads "
    "it: scoring a path sums log-probs across DEPTHS, so this scale decides how "
    "peaked a depth looks and therefore how the budget splits between depth and "
    "breadth. Pass it only for ptes built before the getter existed -- read the "
    "real value off that build's qdq_lm_head.pt2, never another build's.");
DEFINE_string(
    dflash_emb_pte_path,
    "",
    "[DFlash Decoding] Path to the compiled token-embedding pte (headless "
    "decoder consumes embeds, not token ids). "
    "Default: <dflash_draft dir>/tok_embedding_qnn.pte");
DEFINE_string(
    dflash_lm_head_pte_path,
    "",
    "[DFlash Decoding] Path to the compiled lm_head pte (headless decoder emits "
    "hidden, not logits). "
    "Default: <dflash_draft dir>/lm_head_qnn.pte");

std::vector<std::string> CollectPrompts(int argc, char** argv) {
  // Collect all prompts from command line, example usage:
  // --prompt "prompt1" --prompt "prompt2" --prompt "prompt3"
  std::vector<std::string> prompts;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
      prompts.push_back(argv[i + 1]);
      i++; // Skip the next argument
    }
  }
  return prompts;
}

std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::DecoderModelVersion decoder_model_version) {
  std::string formatted_prompt;
  switch (decoder_model_version) {
    case example::DecoderModelVersion::kLlama2:
    case example::DecoderModelVersion::kQwen2_5:
    case example::DecoderModelVersion::kCodegen:
      formatted_prompt.append(prompt);
      break;
    case example::DecoderModelVersion::kLlama3:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      break;
    case example::DecoderModelVersion::kGemma:
    case example::DecoderModelVersion::kGemma3:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<end_of_turn>\n");
      }
      break;
    case example::DecoderModelVersion::kGemma2:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      break;
    case example::DecoderModelVersion::kGranite:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|start_of_role|>system<|end_of_role|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end_of_text|>\n");
      }
      formatted_prompt.append("<|start_of_role|>user<|end_of_role|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end_of_text|>\n");
      formatted_prompt.append("<|start_of_role|>assistant<|end_of_role|>");
      break;
    case example::DecoderModelVersion::kPhi4:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end|>");
      }
      formatted_prompt.append("<|user|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end|><|assistant|>");
      break;
    case example::DecoderModelVersion::kQwen3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      // Qwen3 is a hybrid model: thinking is not a weight, it is a chat-template
      // choice. Non-thinking works by pre-filling an EMPTY think block, which the
      // model reads as "reasoning is already done". Leaving it out is the *thinking*
      // template, and the model duly writes its own <think> essay before answering.
      // Byte-for-byte the same as HF's apply_chat_template(enable_thinking=...).
      if (!FLAGS_enable_thinking) {
        formatted_prompt.append("<think>\n\n</think>\n\n");
      }
      break;
    case example::DecoderModelVersion::kSmollm2_135m:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n\n");
      break;
    case example::DecoderModelVersion::kSmollm3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("\n\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      break;
    case example::DecoderModelVersion::kGlm:
      formatted_prompt.append("<|user|>\n");
      formatted_prompt.append(prompt);
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>\n");
        formatted_prompt.append(system_prompt);
      }
      formatted_prompt.append("<|assistant|>\n");
      break;
    default:
      ET_CHECK_MSG(false, "unsupported llama version");
      break;
  }
  return formatted_prompt;
}

void start_runner(
    std::unique_ptr<executorch::extension::Module> module,
    std::vector<std::string>& prompts,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module,
    std::unique_ptr<executorch::extension::Module> eagle_head_module,
    const std::string& eagle_d2t_path,
    const std::string& eagle_t2d_path,
    const std::string& eagle_embed_path,
    std::unique_ptr<executorch::extension::Module> dflash_draft_module,
    std::unique_ptr<executorch::extension::Module> dflash_emb_module,
    std::unique_ptr<executorch::extension::Module> dflash_lm_head_module) {
  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default ? false
                                                                         : true;
  // create llama runner
  example::Runner runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      nullptr,
      std::move(attention_sink_rope_module),
      std::move(eagle_head_module),
      FLAGS_max_tree_size,
      FLAGS_draft_len,
      FLAGS_tree_depth,
      FLAGS_tree_topk,
      eagle_d2t_path,
      eagle_t2d_path,
      eagle_embed_path,
      std::move(dflash_draft_module),
      FLAGS_block_size,
      FLAGS_dflash_max_context_len,
      std::move(dflash_emb_module),
      std::move(dflash_lm_head_module),
      FLAGS_dflash_tree_budget,
      static_cast<float>(FLAGS_dflash_logit_out_scale));
  auto decoder_model_version = runner.get_decoder_model_version();
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
  std::ofstream fout(FLAGS_output_path.c_str());
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };
  executorch::extension::llm::GenerationConfig config{
      true,
      "",
      "",
      false,
      -1,
      false,
      FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};
  if (use_tokenized_prompt) {
    runner.generate_from_prompt_or_file(
        FLAGS_tokenized_prompt.c_str(), use_tokenized_prompt, config, callback);
  } else {
    // generate tokens & store inference output
    for (int i = 0; i < FLAGS_num_iters; i++) {
      for (const auto& prompt : prompts) {
        std::string formatted_prompt;
        formatted_prompt = get_formatted_prompt(
            prompt, FLAGS_system_prompt, decoder_model_version.get());
        runner.generate_from_prompt_or_file(
            formatted_prompt.c_str(), use_tokenized_prompt, config, callback);
      }
    }
  }

  fout.write(buf.data(), buf.size());
  fout.close();
}

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Only provide prompt or tokenized_input but not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(
        false, "Only TokenGenerator(kv) mode is supported to dump all logits.");
  }

  std::unique_ptr<executorch::extension::Module> module =
      std::make_unique<executorch::extension::Module>(
          FLAGS_model_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;
  if (!FLAGS_attention_sink_rope_path.empty()) {
    attention_sink_rope_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_attention_sink_rope_path.c_str(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
  std::unique_ptr<executorch::extension::Module> eagle_head_module;
  if (!FLAGS_eagle_head_path.empty()) {
    eagle_head_module = std::make_unique<executorch::extension::Module>(
        FLAGS_eagle_head_path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  // Default sibling paths: if not explicitly given, derive from head pte dir.
  auto sibling_path = [](const std::string& head, const std::string& name) {
    if (!head.empty()) {
      auto sep = head.find_last_of("/\\");
      std::string dir =
          (sep != std::string::npos) ? head.substr(0, sep) : ".";
      return dir + "/" + name;
    }
    return std::string{};
  };
  std::string d2t_path = FLAGS_eagle_d2t_path.empty()
      ? sibling_path(FLAGS_eagle_head_path, "d2t.bin")
      : FLAGS_eagle_d2t_path;
  std::string t2d_path = FLAGS_eagle_t2d_path.empty()
      ? sibling_path(FLAGS_eagle_head_path, "t2d.bin")
      : FLAGS_eagle_t2d_path;
  std::string embed_path = FLAGS_eagle_embed_path.empty()
      ? sibling_path(FLAGS_eagle_head_path, "embed.bin")
      : FLAGS_eagle_embed_path;

  std::unique_ptr<executorch::extension::Module> dflash_draft_module;
  std::unique_ptr<executorch::extension::Module> dflash_emb_module;
  std::unique_ptr<executorch::extension::Module> dflash_lm_head_module;
  if (!FLAGS_dflash_draft_path.empty()) {
    dflash_draft_module = std::make_unique<executorch::extension::Module>(
        FLAGS_dflash_draft_path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    // The recompiled decoder is headless: it takes embeds (u16) and emits hidden
    // (u16), so the embedding and lm_head projections run as their own ptes.
    std::string emb_pte = FLAGS_dflash_emb_pte_path.empty()
        ? sibling_path(FLAGS_dflash_draft_path, "tok_embedding_qnn.pte")
        : FLAGS_dflash_emb_pte_path;
    std::string lm_head_pte = FLAGS_dflash_lm_head_pte_path.empty()
        ? sibling_path(FLAGS_dflash_draft_path, "lm_head_qnn.pte")
        : FLAGS_dflash_lm_head_pte_path;
    dflash_emb_module = std::make_unique<executorch::extension::Module>(
        emb_pte.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    dflash_lm_head_module = std::make_unique<executorch::extension::Module>(
        lm_head_pte.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  start_runner(
      std::move(module),
      prompts,
      std::move(attention_sink_rope_module),
      std::move(eagle_head_module),
      d2t_path,
      t2d_path,
      embed_path,
      std::move(dflash_draft_module),
      std::move(dflash_emb_module),
      std::move(dflash_lm_head_module));

  return 0;
}
