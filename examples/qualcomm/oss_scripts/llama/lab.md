#### LLAMA3.2 3B Instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.91.160:34707 -m SM8750 --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --params models/Llama-3.2-1B-Instruct/params.json --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --artifact models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_hybrid_128_1024

python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.91.160:34707 -m SM8750 --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --params models/Llama-3.2-1B-Instruct/params.json --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --pre_gen_pte models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_hybrid_128_1024 --skip_push

python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.191.53:38893 -m SM8750 --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --params models/Llama-3.2-1B-Instruct/params.json --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode hybrid --prefill_ar_len 32 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --pre_gen_pte models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_hybrid_32_1024 --skip_push
```

test spec-bench
```bash
python run_specbench.py \
    --prompt_file /mnt/hdd-ws/users/zqchen/study/device_llm_datasets/Spec-Bench/data/spec_bench/question.jsonl \
    -b build-android -s 10.87.191.53:38893 -m SM8750 \
    --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth \
    --params models/Llama-3.2-1B-Instruct/params.json \
    --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model \
    --decoder_model llama3_2-3b_instruct \
    --model_mode hybrid \
    --prefill_ar_len 32 \
    --max_seq_len 1024 \
    --pre_gen_pte models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_hybrid_32_1024 \
    --skip_push \
    --output_dir outputs \
    --limit 5
```

lookahead
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.61.11:41153 -m SM8750 --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --params models/Llama-3.2-1B-Instruct/params.json --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode lookahead --prefill_ar_len 32 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --ngram 3 --window 2 --gcap 2 --artifact models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_lookahead_32_1024_3_2_2

python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.61.11:41153 -m SM8750 --checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --params models/Llama-3.2-1B-Instruct/params.json --tokenizer_model models/Llama-3.2-1B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode lookahead --prefill_ar_len 32 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --ngram 3 --window 2 --gcap 2 --pre_gen_pte models/Llama-3.2-1B-Instruct/llama3_2-3b_instruct_lookahead_32_1024_3_2_2 --skip_push
```

draft mode 
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.166.27:34113 -m SM8750 --checkpoint models/Llama-3.2-3B-Instruct/consolidated.00.pth --params models/Llama-3.2-3B-Instruct/params.json --tokenizer_model models/Llama-3.2-3B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode draft --draft_decoder_model llama3_2-1b_instruct --draft_len 4 --draft_model_checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --draft_model_params models/Llama-3.2-1B-Instruct/params.json --prefill_ar_len 32 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?"

python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s 10.87.52.104:37011 -m SM8750 --checkpoint models/Llama-3.2-3B-Instruct/consolidated.00.pth --params models/Llama-3.2-3B-Instruct/params.json --tokenizer_model models/Llama-3.2-3B-Instruct/tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode draft --draft_decoder_model llama3_2-1b_instruct --draft_len 4 --draft_model_checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth --draft_model_params models/Llama-3.2-1B-Instruct/params.json --prefill_ar_len 32 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --pre_gen_pte models/Llama-3.2-3B-Instruct/llama3_2-3b_instruct_draft_32_1024_4 --skip_push
```


现在我对整个流程有了完整的了解。让我整理一个完整的实验指南。
                                                                             
  ---                                                                                                                                           
  整体流程
                                                                                                                                                
  DraftManager 会同时编译两个模型：
  - 目标模型（3B）：走 HybridTextDecoder → QNN NPU .pte                                                                                         
  - 草稿模型（1B）：走 DraftDecoder → CPU fp32 .pte                                                                                             
                                                                                                                                                
  --decoder_model 指定目标模型的配置，--draft_model_checkpoint / --draft_model_params 指定草稿模型的权重和架构。                                
                                                                                                                                                
  ---                                                                                                                                           
  Step 1：准备模型权重                                                                                                                          
                                                                                                                                                
  Llama 3.2 的 repo_id = None，必须提供本地 checkpoint。有两种来源：
                                                                                                                                                
  方式 A：从 Meta 官方下载（原生格式）                                                                                                          
                                                                                                                                                
  ├── Llama-3.2-1B-Instruct/                                                                                                                    
  │   ├── consolidated.00.pth                                                                                                                   
  │   ├── params.json                                                                                                                           
  │   └── tokenizer.model                                                                                                                       
  └── Llama-3.2-3B-Instruct/                                                                                                                    
      ├── consolidated.00.pth   # 3B 单分片或多分片                                                                                             
      ├── params.json                                                                                                                           
      └── tokenizer.model                                                                                                                       
                                                                                                                                                
  方式 B：从 HuggingFace 下载并转换                                                                                                             
                                                                                                                                                
  # 安装依赖                                                                                                                                    
  pip install huggingface_hub                                                                                                                   
                  
  # 下载 1B                                                                                                                                     
  python -c "
  from huggingface_hub import snapshot_download                                                                                                 
  snapshot_download('meta-llama/Llama-3.2-1B-Instruct', local_dir='models/Llama-3.2-1B-Instruct')
  "                                                                                                                                             
                  
  # 下载 3B                                                                                                                                     
  python -c "     
  snapshot_download('meta-llama/Llama-3.2-3B-Instruct', local_dir='models/Llama-3.2-3B-Instruct')
  "                                                                                              
                                                                                                                                                
  # HuggingFace 格式需要转换成 Meta 格式（consolidated.pth）                                                                                    
  # 使用 executorch 自带的转换脚本                                                                                                              
  python examples/models/llama/convert_weights.py \                                                                                             
    --input_dir models/Llama-3.2-1B-Instruct \                                                                                                  
    --output_dir models/Llama-3.2-1B-Instruct \                                                                                                 
    --model llama3                                                                                                                              
                                                                                                                                                
  ---                                                                                                                                           
  Step 2：编译两个 PTE                                                                                                                          
                                                                                                                                                
  python examples/qualcomm/oss_scripts/llama/llama.py \
    -b build-android \                                                                                                                          
    -m SM8750 \   
    --decoder_model llama3_2-3b_instruct \                                                                                                      
    --checkpoint models/Llama-3.2-3B-Instruct/consolidated.00.pth \                                                                             
    --params models/Llama-3.2-3B-Instruct/params.json \                                                                                         
    --tokenizer_model models/Llama-3.2-3B-Instruct/tokenizer.model \                                                                            
    --model_mode draft \                                                                                                                        
    --prefill_ar_len 128 \                                                                                                                      
    --max_seq_len 512 \                                                                                                                         
    --max_context_len 512 \                                                                                                                     
    --draft_len 4 \                                                                                                                             
    --draft_model_checkpoint models/Llama-3.2-1B-Instruct/consolidated.00.pth \                                                                 
    --draft_model_params models/Llama-3.2-1B-Instruct/params.json \                                                                             
    --prompt "Tell me about speculative decoding." \                                                                                            
    --artifact models/draft_output \                                                                                                            
    --compile_only                                                                                                                              
                                                                                                                                                
  产出：                                                                                                                                        
  models/draft_output/                                                                                                                          
  ├── draft_llama_qnn.pte   # 3B 目标模型，QNN NPU                                                                                              
  └── draft_model.pte       # 1B 草稿模型，CPU fp32                                                                                             
                                                                                                                                                
  ---                                                                                                                                           
  Step 3：推理                                                                                                                                  
                                                                                                                                                
  python examples/qualcomm/oss_scripts/llama/llama.py \
    -b build-android \                                                                                                                          
    -s ${SERIAL_NUM} \                                                                                                                          
    -m SM8750 \                                                                                                                                 
    --decoder_model llama3_2-3b_instruct \                                                                                                      
    --checkpoint models/Llama-3.2-3B-Instruct/consolidated.00.pth \                                                                             
    --params models/Llama-3.2-3B-Instruct/params.json \                                                                                         
    --tokenizer_model models/Llama-3.2-3B-Instruct/tokenizer.model \                                                                            
    --model_mode draft \                                                                                                                        
    --prefill_ar_len 128 \                                                                                                                      
    --max_seq_len 512 \                                                                                                                         
    --max_context_len 512 \                                                                                                                     
    --draft_len 4 \                                                                                                                             
    --prompt "Tell me about speculative decoding." \                                                                                            
    --pre_gen_pte models/draft_output                                                                                                           
                                                                                                                                                
  ---                                                                                                                                           
  注意事项                                                                                                                                      
                                                                                                                                                
  3B 多分片 checkpoint：如果 3B 下载后是多个分片（consolidated.00.pth, consolidated.01.pth），需要先合并：
  import torch, glob                                                                                                                            
  shards = sorted(glob.glob("models/Llama-3.2-3B-Instruct/consolidated.*.pth"))
  state_dict = {}                                                                                                                               
  for s in shards:
      state_dict.update(torch.load(s, map_location="cpu", weights_only=True))
  torch.save(state_dict, "models/Llama-3.2-3B-Instruct/consolidated.00.pth")                                                                    
                                                                                                                                                
  --draft_model_params 必须提供：1B 和 3B 的 params.json 架构不同（层数、头数），不提供会导致草稿模型用 3B 的架构加载 1B 的权重，直接报错。     
                                                                                                                                                
  --decoder_model 始终指向目标模型：这个参数决定 QNN 编译配置（量化方案、分片数等），草稿模型的架构完全由 --draft_model_params 决定。           
    --pre_gen_pte models/draft_output

  ---
  注意事项

  3B 多分片 checkpoint：如果 3B 下载后是多个分片（consolidated.00.pth, consolidated.01.pth），需要先合并：
  import torch, glob
  shards = sorted(glob.glob("models/Llama-3.2-3B-Instruct/consolidated.*.pth"))
  state_dict = {}
  for s in shards:
      state_dict.update(torch.load(s, map_location="cpu", weights_only=True))
  torch.save(state_dict, "models/Llama-3.2-3B-Instruct/consolidated.00.pth")

  --draft_model_params 必须提供：1B 和 3B 的 params.json 架构不同（层数、头数），不提供会导致草稿模型用 3B 的架构加载 1B 的权重，直接报错。

  --decoder_model 始终指向目标模型：这个参数决定 QNN 编译配置（量化方案、分片数等），草稿模型的架构完全由 --draft_model_params 决定。

  接受率日志：运行时 C++ 端会打印：
  Speculative Decoding: n_generated=X, n_draft=Y, n_accepted=Z, accept_rate=W%
  1B→3B 的接受率通常在 60-80%，取决于任务类型。

✻ Cogitated for 8m 42s