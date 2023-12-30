from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    mu = shared.args.multi_user
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("Parameters", elem_id="parameters"):
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='Preset', elem_classes='slim-dropdown', info='パラメータの組み合わせを保存およびロードして再利用するために使用できます')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu, info='保存')
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu, info='削除')
                        shared.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button', info='ランダム')

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label="Filter by loader", choices=["All"] + list(loaders.loaders_and_params.keys()), value="All", elem_classes='slim-dropdown', info='')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'], info='生成するトークンの最大数。必要以上に高く設定しないでください。これは、数式による切り捨て計算で使用される(prompt_length) = min(truncation_length - max_new_tokens, prompt_length)ため、高く設定しすぎるとプロンプトが切り捨てられます')
                            shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature', info='出力のランダム性を制御する主な要素。0 = 決定的 (最も可能性の高いトークンのみが使用されます)。値が大きいほどランダム性が高くなります')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p', info='1 に設定されていない場合は、合計がこの数値未満になる確率を持つトークンを選択します。値が高いほど、考えられるランダムな結果の範囲が広がります')
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='min_p', info='より小さい確率のトークンは(min_p) * (probability of the most likely token)破棄されます。これは top_a と同じですが、確率を 2 乗しません。')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k', info='top_p と似ていますが、代わりに、最も可能性の高い top_k トークンのみを選択します。値が高いほど、考えられるランダムな結果の範囲が広がります')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty', info='前のトークンを繰り返す場合のペナルティ係数。1 はペナルティがないことを意味し、値が高いほど繰り返しが少なくなり、値が低いほど繰り返しが多くなります')
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='presence_penalty', info='repetition_penalty に似ていますが、乗算係数の代わりに生のトークン スコアに加算オフセットを使用します。より良い結果が得られる可能性があります。0 はペナルティがないことを意味し、値が大きい = 繰り返しが少なく、値が低い = 繰り返しが多くなります。以前は「additive_repetition_penalty」と呼ばれていました')
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='frequency_penalty', info='コンテキスト内でトークンが出現した回数に基づいて調整される反復ペナルティ。これには注意してください。トークンに課せられるペナルティの量に制限はありません')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range', info='反復ペナルティの対象となる最新のトークンの数。0 を指定すると、すべてのトークンが使用されます')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p', info='')
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs', info='ディストリビューション内で確率の低いトークンの末尾の検出を試み、それらのトークンを削除します。詳細については、ブログ投稿を参照してください。0 に近づくほど、破棄されるトークンが多くなります')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a', info='より小さい確率のトークンは(top_a) * (probability of the most likely token)^2破棄されます')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff', info='1e-4 の単位。適切な値は 3 です。これにより、トークンがサンプリングから除外される確率の下限が設定されます')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff', info='1e-4 の単位。適切な値は 3 です。特別な Eta サンプリング手法の主要なパラメータです。説明については、この文書を参照してください')

                        with gr.Column():
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='Classifier-Free Guide (CFG) の主要パラメータ。論文では、 1.5 が適切な値であると示唆しています。否定プロンプトと組み合わせて使用​​することも、そうでないこともできます')
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='Negative prompt', lines=3, elem_classes=['add_scrollbar'], info='guidance_scale != 1 の場合にのみ使用されます。これは、モデルの指示やカスタム システム メッセージに最も役立ちます。このフィールドに完全なプロンプトを配置し、システム メッセージをモデルのデフォルトのメッセージ (「あなたはラマ、役に立つアシスタントです...」など) に置き換えて、モデルがカスタム システム メッセージにさらに注意を払うようにします')
                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='これをゼロより大きく設定し、「do_sample」のチェックを外すと、対照検索が有効になります。これは、top_k の低い値 (たとえば、top_k = 4) で使用する必要があります')
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='Mirostat サンプリング手法を有効にします。サンプリング中の混乱を制御することを目的としています。論文を参照してください')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau', info='わかりません。詳細については論文を参照してください。Preset Arena によれば、8 が適切な値です')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta', info='わかりません。詳細については論文を参照してください。Preset Arena によると、0.1 が適切な値です')
                            shared.gradio['temperature_last'] = gr.Checkbox(value=generate_params['temperature_last'], label='temperature_last', info='温度を最初ではなく最後のサンプラーにします。これにより、min_p のようなサンプラーで確率の低いトークンを削除し、高温を使用して一貫性を失うことなくモデルをクリエイティブにすることができます')
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample', info='チェックを外すと、サンプリングが完全に無効になり、代わりに貪欲なデコードが使用されます (最も可能性の高いトークンが常に選択されます)。')
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)', info='Pytorch シードをこの番号に設定します。一部のローダーは Pytorch を使用せず (特に llama.cpp)、その他は決定論的ではない (特に ExLlama v1 および v2) ことに注意してください。これらのローダーの場合、シードは効果がありません')
                            with gr.Accordion('Other parameters', open=False):
                                shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty', info='「幻覚フィルター」とも呼ばれます。前のテキストにないトークンにペナルティを与えるために使用されます。値が高いほどコンテキスト内にとどまる可能性が高く、値が低いほど発散する可能性が高くなります')
                                shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size', info='0 に設定されていない場合、反復がまったくブロックされるトークン セットの長さを指定します。値が高いほど大きなフレーズがブロックされ、値が低いほど単語や文字の繰り返しがブロックされます。ほとんどの場合、0 または高い値のみを使用することをお勧めします')
                                shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length', info=' トークンの最小生成長。これは、トランスフォーマー ライブラリの組み込みパラメータですが、あまり役に立ちませんでした。通常は、代わりに「eos_token を禁止する」にチェックを入れます')
                                shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams', info='ビーム検索のビーム数。1 はビームサーチを行わないことを意味します')
                                shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty', info='ビーム検索でのみ使用されます。length_penalty > 0.0より長いシーケンスを促進する一方、length_penalty < 0.0より短いシーケンスを奨励します。')
                                shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping', info='ビーム検索でのみ使用されます。チェックすると、「num_beams」個の完全な候補が存在するとすぐに生成が停止します。それ以外の場合は、ヒューリスティックが適用され、より良い候補が見つかる可能性が非常に低いときに生成が停止します (これをトランスのドキュメントからコピーしただけで、良好な結果を生成するためのビーム検索は一度も行っていません)')

                    gr.Markdown("[Learn more](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab)")

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Slider(value=get_truncation_length(), minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=256, label='Truncate the prompt up to this length', info='プロンプトがこの長さを超える場合、左端のトークンが削除されます。 ほとんどのモデルでは、これが最大 2048 である必要があります プロンプトがモデルのコンテキストの長さより大きくならないようにするために使用されます。メモリを動的に割り当てるトランスフォーマー ローダーの場合、このパラメータを使用して VRAM 上限を設定し、メモリ不足エラーを防ぐこともできます。このパラメーターは、モデルをロードすると、モデルのコンテキスト長 (これらのパラメーターを使用するローダーの場合は「n_ctx」または「max_seq_len」から、使用しないローダーの場合はモデル メタデータから直接) で自動的に更新されます')
                            shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='Maximum tokens/second', info='モデルの生成が速すぎる場合に、テキストをリアルタイムで読み取れるようにします。柔軟に自分の GPU がいかに優れているかをみんなに伝えたい場合に最適です')
                            shared.gradio['max_updates_second'] = gr.Slider(value=shared.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='Maximum UI updates/second', info='ストリーミング中に UI に遅延が発生する場合は、これを設定します')

                            shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='デフォルトに加えて。 "" の間にカンマで区切って書きます このフィールドに設定された文字列のいずれかが生成されると、モデルはすぐに生成を停止します。[チャット] タブでテキストを生成する場合、このパラメータに関係なく、チャット モードの「\nYour Name:」や「\nBot name:」など、いくつかのデフォルトの停止文字列が設定されることに注意してください。このパラメータの名前に「Custom」が付いているのはそのためです', placeholder='"\\n", "\\nYou:" ')
                            shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label='Custom token bans', info='生成を禁止する特定のトークン ID をカンマで区切って指定します。 ID は [デフォルト] タブまたは [ノートブック] タブで確認できます モデルによる特定のトークンの生成を完全に禁止できます。tokenizer.jsonトークン ID は、「デフォルト」 > 「トークン」または「ノートブック」 > 「トークン」で見つけるか、モデルの を直接調べる必要があります')

                        with gr.Column():
                            shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='max_new_tokens を利用可能なコンテキストの長さまで拡張します パラメーターはバックエンドで利用可能なコンテキストの長さまで拡張されます。最大長は「truncation_length」パラメータで指定されます。これは、[続行] を何度もクリックすることなく、[チャット] タブで長い返信を得るのに便利です')
                            shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='モデルが途中で世代を終了しないように強制します モデルが生成できるトークンの 1 つは、EOS (End of Sequence) トークンです。生成されると、生成は途中で停止します。このパラメーターがチェックされている場合、そのトークンの生成は禁止され、生成では常に「max_new_tokens」トークンが生成されます')
                            shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='これを無効にすると、返信がよりクリエイティブになる可能性があります デフォルトでは、トークナイザーは BOS (Beginning of Sequence) トークンをプロンプトに追加します。トレーニング中、BOS トークンはさまざまなドキュメントを分離するために使用されます。チェックを外した場合、BOS トークンは追加されず、モデルはプロンプトが文書の先頭ではなく文書の途中にあるものとして解釈します。これにより出力が大幅に変わり、より創造的なものになります')
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='一部の特定のモデルでは、この設定を解除する必要があります 生成されたトークンをデコードするときに、特殊トークンがテキスト表現に変換されるのをスキップします。それ以外の場合、BOS は<s>、EOS は</s>などと表示されます')
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='Activate text streaming', info='チェックを外すと、単語を 1 つずつストリーミングせずに、完全な応答が一度に出力されます。Google Colab で WebUI を実行したり、 を使用したりするなど、待ち時間の長いネットワークでは、このパラメーターのチェックを外すことをお勧めします')

                    with gr.Row() as shared.gradio['grammar_file_row']:
                            shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label='Load grammar from file (.gbnf)', elem_classes='slim-dropdown', info='text-generation-webui/grammars の下のファイルから GBNF 文法をロードします。出力は下の「文法」ボックスに書き込まれます。このメニューを使用してカスタム文法を保存および削除することもできます ')
                            ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button', interactive=not mu)
                            shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu, info='保存')
                            shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu, info='削除')

                    gr.Markdown("詳細については、[GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) ガイドを参照してください")
                    shared.gradio['grammar_string'] = gr.Textbox(value='', label='Grammar', lines=16, elem_classes=['add_scrollbar', 'monospace'], info='モデルの出力を特定の形式に制限できます。たとえば、モデルにリスト、JSON、特定の単語などを生成させることができます。Grammar は非常に強力なので、強くお勧めします。構文は一見すると少し難しそうに見えますが、理解すれば非常に簡単になります')

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['random_preset'].click(presets.random_preset, gradio('interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'))


def get_truncation_length():
    if 'max_seq_len' in shared.provided_arguments or shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.provided_arguments or shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r').read()
    else:
        return ''
