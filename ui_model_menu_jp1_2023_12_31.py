import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_xpu_available

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_model_settings,
    update_model_parameters
)
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # Finding the default values for the GPU and CPU memories
    total_mem = []
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
    else:
        for i in range(torch.cuda.device_count()):
            total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)

    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Tab("Model", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='Model', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                            shared.gradio['load_model'] = gr.Button("Load", visible=not shared.settings['autoload_model'], elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['unload_model'] = gr.Button("Unload", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['reload_model'] = gr.Button("Reload", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['save_model_settings'] = gr.Button("Save settings", elem_classes='refresh-button', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                            shared.gradio['lora_menu_apply'] = gr.Button(value='Apply LoRAs', elem_classes='refresh-button', interactive=not mu)

        with gr.Row():
            with gr.Column():
                shared.gradio['loader'] = gr.Dropdown(label="Model loader", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            for i in range(len(total_mem)):
                                shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i], info='0 より大きく設定すると、アクセラレータ ライブラリを使用した CPU オフロードが有効になり、レイヤーの一部が CPU に送られます。パフォーマンスは非常に悪いです。アクセラレータはこのパラメータを文字通りに扱うわけではないので、VRAM 使用量を最大 10 GiB にしたい場合は、このパラメータを 9 GiB または 8 GiB に設定する必要がある場合があることに注意してください。私の知る限り、「load_in_8bit」と組み合わせて使用​​することはできますが、「load-in-4bit」と組み合わせて使用​​することはできません')

                            shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem, info='上記のパラメータと同様に、使用される CPU メモリの量に制限を設定することもできます。GPU または CPU のどちらにも適合しないものはディスク キャッシュに移動されるため、このオプションを使用するには、「ディスク」チェックボックスもオンにする必要があります')
                            shared.gradio['transformers_info'] = gr.Markdown('load-in-4bit params:')
                            shared.gradio['compute_dtype'] = gr.Dropdown(label="compute_dtype", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype, info='デフォルト値のままにすることをお勧めします')
                            shared.gradio['quant_type'] = gr.Dropdown(label="quant_type", choices=["nf4", "fp4"], value=shared.args.quant_type, info='デフォルト値のままにすることをお勧めします')
                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq_backend", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)

                            shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=128, value=shared.args.n_gpu_layers, info='GPU に割り当てるレイヤーの数。0 に設定すると、CPU のみが使用されます。すべてのレイヤーをオフロードしたい場合は、これを最大値に設定するだけです')
                            shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=shared.settings['truncation_length_max'], step=256, label="n_ctx", value=shared.args.n_ctx, info='コンテキストの長さ。 モデルのロード中にメモリが不足した場合は、この値を下げてみてください。')
                            shared.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=32, value=shared.args.threads, info='推奨値: 物理コアの数')
                            shared.gradio['threads_batch'] = gr.Slider(label="threads_batch", minimum=0, step=1, maximum=32, value=shared.args.threads_batch, info='推奨値: コアの合計数 (物理 + 仮想)')
                            shared.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, value=shared.args.n_batch, info='プロンプト処理のバッチ サイズ。値を大きくすると生成が速くなると考えられていますが、この値を変更しても何のメリットも得られませんでした')

                            shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None" , info='適切なメタデータのない古いモデルの場合、モデルの精度をビット単位で手動で設定します。通常は無視できます')
                            shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None", info='適切なメタデータのない古代モデルの場合、モデル グループ サイズを手動で設定します。通常は無視できます')
                            shared.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None"], value=shared.args.model_type or "None", info='')
                            shared.gradio['pre_layer'] = gr.Slider(label="pre_layer", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0, info='')
                            shared.gradio['autogptq_info'] = gr.Markdown('* Llama から派生したモデルの場合は、AutoGPTQ よりも ExLlama_HF を推奨します')
                            shared.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='GPU ごとに使用する VRAM (GB 単位) のカンマ区切りのリスト。 例: 20,7,7')
                            shared.gradio['max_seq_len'] = gr.Slider(label='max_seq_len', minimum=0, maximum=shared.settings['truncation_length_max'], step=256, info='コンテキストの長さ。 モデルのロード中にメモリが不足した場合は、この値を下げてみてください。', value=shared.args.max_seq_len)
                            shared.gradio['alpha_value'] = gr.Slider(label='alpha_value', minimum=1, maximum=8, step=0.05, info='NTK RoPE スケーリングの位置埋め込みアルファ係数。 推奨値（NTKv1）：1.5x コンテキストの場合は 1.75、2x コンテキストの場合は 2.5。 これと compress_pos_emb の両方ではなく、どちらかを使用してください。', value=shared.args.alpha_value)
                            shared.gradio['rope_freq_base'] = gr.Slider(label='rope_freq_base', minimum=0, maximum=1000000, step=1000, info='0 より大きい場合は、alpha_value の代わりに使用されます。 これら 2 つは、rope_freq_base = 10000 * alpha_value ^ (64 / 63) によって関連付けられます。', value=shared.args.rope_freq_base)
                            shared.gradio['compress_pos_emb'] = gr.Slider(label='compress_pos_emb', minimum=1, maximum=8, step=1, info='位置埋め込み圧縮係数。 (コンテキストの長さ) / (モデルの元のコンテキストの長さ) に設定する必要があります。 1/rope_freq_scale に等しい。', value=shared.args.compress_pos_emb)
                            shared.gradio['quipsharp_info'] = gr.Markdown('QuIP# only works on Linux.')

                        with gr.Column():
                            shared.gradio['tensorcores'] = gr.Checkbox(label="tensorcores", value=shared.args.tensorcores, info='tensor コアのサポートでコンパイルされた llama-cpp-python を使用します。 これにより、RTX カードのパフォーマンスが向上します。 NVIDIAのみ。')
                            shared.gradio['no_offload_kqv'] = gr.Checkbox(label="no_offload_kqv", value=shared.args.no_offload_kqv, info='K、Q、V を GPU にオフロードしないでください。 これにより VRAM は節約されますが、パフォーマンスは低下します。')
                            shared.gradio['triton'] = gr.Checkbox(label="triton", value=shared.args.triton, info='Linux でのみ利用可能です。act-order と groupsize の両方を備えたモデルを同時に使用するために必要です。ExLlama は、triton なしでも Windows にこれらの同じモデルをロードできることに注意してください')
                            shared.gradio['no_inject_fused_attention'] = gr.Checkbox(label="no_inject_fused_attention", value=shared.args.no_inject_fused_attention, info='Disable fused attention. Fused attention improves inference performance but uses more VRAM. Fuses layers for AutoAWQ. Disable if running low on VRAM.融合された注意を無効にします。 融合された注意により推論パフォーマンスは向上しますが、より多くの VRAM を使用します。 AutoAWQ のレイヤーを融合します。 VRAM が不足している場合は無効にします。')
                            shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="no_inject_fused_mlp", value=shared.args.no_inject_fused_mlp, info='Affects Triton only. Disable fused MLP. Fused MLP improves performance but uses more VRAM. Disable if running low on VRAM.トリトンのみに影響します。 融合 MLP を無効にします。 融合 MLP はパフォーマンスを向上させますが、より多くの VRAM を使用します。 VRAM が不足している場合は無効にします。')
                            shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="no_use_cuda_fp16", value=shared.args.no_use_cuda_fp16, info='これにより、一部のシステムではモデルが高速化される可能性があります。これを設定しないとパフォーマンスが非常に低下する可能性があります。通常は無視できます')
                            shared.gradio['desc_act'] = gr.Checkbox(label="desc_act", value=shared.args.desc_act, info='「desc_act」、「wbits」、「groupsize」は、quantize_config.json のない古いモデルに使用されます 適切なメタデータのない古代モデルの場合、モデルの「act-order」パラメーターを手動で設定します。通常は無視できます')
                            shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="no_mul_mat_q", value=shared.args.no_mul_mat_q, info='mul_mat_q カーネルを無効にします。通常、このカーネルにより生成速度が大幅に向上します。一部のシステムで機能しない場合に備えて、これを無効にするこのオプションが含まれています')
                            shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap, info='モデルを一度にメモリにロードします。ロード時間が長くなり、後で I/O 操作ができなくなる可能性があります')
                            shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock, info='スワップや圧縮ではなく、モデルを RAM に保持するようにシステムに強制します (これが何を意味するのかわかりません。一度も使用したことがありません)')
                            shared.gradio['numa'] = gr.Checkbox(label="numa", value=shared.args.numa, info='NUMA サポートは、不均一なメモリ アクセスを持つ一部のシステムで役立ちます。')
                            shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu, info='GPU アクセラレーションなしでコンパイルされた llama.cpp のバージョンを強制的に使用します。通常は無視できます。CPU のみを使用する場合にのみこれを設定し、それ以外の場合は llama.cpp が機能しません。')
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit, info='bitsandbytes を使用して 8 ビット精度でモデルをロードします。このライブラリの 8 ビット カーネルは、推論ではなくトレーニング用に最適化されているため、8 ビットでのロードは 4 ビットでのロードよりも遅くなります (ただし、精度は高くなります)。')
                            shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16, info='float16 (デフォルト) の代わりに bfloat16 精度を使用します。量子化が使用されていない場合にのみ適用されます')
                            shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices, info='チェックすると、バックエンドは CPU オフロードでモデルをロードできるように、「gpu-memory」の適切な値を推測しようとします。代わりに「gpu-memory」を手動で設定することをお勧めします。このパラメーターは GPTQ モデルをロードする場合にも必要です。その場合、モデルをロードする前にチェックする必要があります。')
                            shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk, info='GPU と CPU の組み合わせに適合しないレイヤーのディスク オフロードを有効にします。')
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=shared.args.load_in_4bit, info='bitsandbytes を使用して 4 ビット精度でモデルをロードします')
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=shared.args.use_double_quant)
                            shared.gradio['tensor_split'] = gr.Textbox(label='tensor_split', info='マルチ GPU のみ。GPU ごとに割り当てるメモリの量を設定します モデルを複数の GPU、コンマ区切りの比率リストに分割します。 18,17')
                            shared.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=shared.args.trust_remote_code, info='このオプションを有効にするには、 --trust-remote-code フラグを指定して Web UI を起動します。 一部の機種では必要となります.', interactive=shared.args.trust_remote_code)
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="cfg-cache", value=shared.args.cfg_cache, info='CFG ネガティブ プロンプト用に追加のキャッシュを作成します。「パラメータ」>「生成」タブで CFG を使用する場合にのみ設定する必要があります。このパラメータをチェックすると、キャッシュ VRAM の使用量が 2 倍になります')
                            shared.gradio['logits_all'] = gr.Checkbox(label="logits_all", value=shared.args.logits_all, info='Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower.複雑性の評価が機能するには設定する必要があります。 それ以外の場合は、プロンプトの処理が遅くなるため無視してください。')
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="use_flash_attention_2", value=shared.args.use_flash_attention_2, info='モデルのロード中に use_flash_attention_2=True を設定します。')
                            shared.gradio['disable_exllama'] = gr.Checkbox(label="disable_exllama", value=shared.args.disable_exllama, info='ExLlama カーネルを無効にします。')
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="disable_exllamav2", value=shared.args.disable_exllamav2, info='ExLlamav2 カーネルを無効にします。')
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="no_flash_attn", value=shared.args.no_flash_attn, info='Force flash-attention to not be used.フラッシュアテンションが使用されないよう強制します。')
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="cache_8bit", value=shared.args.cache_8bit, info='8 ビット キャッシュを使用して VRAM を節約します。')
                            shared.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast", value=shared.args.no_use_fast, info='トークナイザーのロード中に use_fast=False を設定します。')
                            shared.gradio['num_experts_per_token'] = gr.Number(label="Number of experts per token", value=shared.args.num_experts_per_token, info='Mixtral などの MoE モデルにのみ適用されます。')
                            shared.gradio['gptq_for_llama_info'] = gr.Markdown('古い GPU との互換性のためのレガシー ローダー。 GPTQ モデルがサポートされている場合は、ExLlama_HF または AutoGPTQ が優先されます。')
                            shared.gradio['exllama_info'] = gr.Markdown("拡張機能との統合を強化し、ローダー間でサンプリング動作の一貫性を高めるために、ExLlama よりも ExLlama_HF を推奨します。")
                            shared.gradio['exllamav2_info'] = gr.Markdown("拡張機能との統合を強化し、ローダー間でサンプリング動作の一貫性を高めるために、ExLlamav2 よりも ExLlamav2_HF を推奨します。")
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown('llamacpp_HF は、llama.cpp をトランスフォーマー モデルとしてロードします。 使用するには、トークナイザーをダウンロードする必要があります。\n\nオプション 1 (推奨): .gguf を、special_tokens_map.json、tokenizer_config.json、tokenizer.json、tokenizer.model の 4 つのファイルとともに models/ のサブフォルダーに配置します。\n\nオプション 2: 「モデルまたは LoRA のダウンロード」の下で「oababooga/llama-tokenizer」をダウンロードします。 これは、一部の (すべてではありません) モデルで機能するデフォルトの Llama トークナイザーです。')


            with gr.Column():
                with gr.Row():
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Autoload the model', info='[モデル] ドロップダウンでモデルを選択したらすぐにモデルをロードするかどうか。', interactive=not mu)

                shared.gradio['custom_model_menu'] = gr.Textbox(label="Download model or LoRA", info="Hugging Face のユーザー名/モデルのパスを入力します (例: facebook/galactica-125m)。 ブランチを指定するには、facebook/gaoptica-125m:main のように、末尾の \":\" 文字の後に追加します。 単一のファイルをダウンロードするには、2 番目のボックスにファイル名を入力します。", interactive=not mu)
                shared.gradio['download_specific_file'] = gr.Textbox(placeholder="File name (for GGUF models)", show_label=False, max_lines=1, interactive=not mu)
                with gr.Row():
                    shared.gradio['download_model_button'] = gr.Button("Download", variant='primary', interactive=not mu)
                    shared.gradio['get_file_list'] = gr.Button("Get file list", interactive=not mu)

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')


def create_event_handlers():
    shared.gradio['loader'].change(
        loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params())).then(
        lambda value: gr.update(choices=loaders.get_model_types(value)), gradio('loader'), gradio('model_type'))

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('model_menu', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['reload_model'].click(
        unload_model, None, None).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['unload_model'].click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, gradio('model_status'))

    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), gradio('autoload_model'), gradio('load_model'))


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f"The settings for `{selected_model}` have been updated.\n\nClick on \"Load\" to load it."
        return

    if selected_model == 'None':
        yield "モデルが選択されていません"
    else:
        try:
            yield f"読み込み中 `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"正常にロードされました `{selected_model}`."

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\nIt seems to be an instruction-following model with template "{}". In the chat tab, instruct or chat-instruct modes should be used.'.format(settings['instruction_template'])

                yield output
            else:
                yield f"読み込みに失敗 `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error('モデルのロードに失敗しました')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("次の LoRA を適用する {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("LoRAx の適用に成功しました")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        progress(0.0)
        downloader = importlib.import_module("download-model").ModelDownloader()
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)
        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            yield '\n\n'.join([f"`{Path(link).name}`" for link in links])
            return

        yield ("出力フォルダーの取得")
        base_folder = shared.args.lora_dir if is_lora else shared.args.model_dir
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp, base_folder=base_folder)
        if check:
            progress(0.5)
            yield ("以前にダウンロードしたファイルを確認する")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}/`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)
            yield ("Done!完了")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
            return state['n_ctx']

    return current_length
