#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 💬 AksaraLLM Chat Demo — Gradio Web Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loads a from-scratch AksaraLLM checkpoint exported to standard HF format via
aksarallm.hf_export (see upload_to_hf.py) — not a fine-tune of another base
model. There's no bundled default repo here because none is published yet
(see the project roadmap); point --model at your own export.

pip install gradio transformers torch
python3 gradio_chat.py --model /path/to/hf_export_dir
python3 gradio_chat.py --model AksaraLLM/your-exported-repo
→ Buka http://localhost:7860
"""

import argparse
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF repo id or local dir produced by aksarallm.hf_export")
ap.add_argument("--port", type=int, default=7860)
cli_args = ap.parse_args()

# ====================================================================
#  CONFIG
# ====================================================================
MODEL_NAME = cli_args.model
SYSTEM_PROMPT = "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu. Jawab pertanyaan dengan akurat dan detail."

# ====================================================================
#  LOAD MODEL
# ====================================================================
print("🔄 Loading AksaraLLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)
model.eval()
print(f"✅ Model loaded! ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

# ====================================================================
#  CHAT FUNCTION
# ====================================================================
def chat(message, history, system_prompt, temperature, max_tokens, top_p):
    """Generate response from AksaraLLM."""
    
    # Build messages from history
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})
    
    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return response

# ====================================================================
#  GRADIO INTERFACE
# ====================================================================
EXAMPLES = [
    ["Siapa kamu?"],
    ["Apa itu Pancasila? Jelaskan sila-silanya!"],
    ["Buatkan fungsi Python untuk menghitung faktorial"],
    ["Jelaskan perbedaan AI dan Machine Learning"],
    ["Ceritakan sejarah kemerdekaan Indonesia"],
    ["Buatkan puisi pendek tentang Indonesia"],
    ["Berapa 17 x 23? Jelaskan langkah-langkahnya"],
    ["Apa ibukota semua provinsi di Pulau Jawa?"],
]

CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="AksaraLLM Chat", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>🇮🇩 AksaraLLM Chat</h1>
        <p style="color: #666; font-size: 16px;">
            Model Bahasa AI Open-Source Indonesia — dilatih dari nol
        </p>
    </div>
    """)

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        bubble_full_width=False,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ketik pesan di sini... (contoh: Apa itu Pancasila?)",
            show_label=False,
            scale=9,
            container=False,
        )
        submit_btn = gr.Button("Kirim 🚀", scale=1, variant="primary")
    
    with gr.Accordion("⚙️ Pengaturan", open=False):
        with gr.Row():
            system_prompt = gr.Textbox(
                value=SYSTEM_PROMPT,
                label="System Prompt",
                lines=2
            )
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=msg,
        label="💡 Contoh Pertanyaan"
    )
    
    gr.HTML(f"""
    <div style="text-align: center; padding: 10px; color: #999; font-size: 12px;">
        <p>AksaraLLM • Apache 2.0 • <a href="https://huggingface.co/{MODEL_NAME}">{MODEL_NAME}</a></p>
    </div>
    """)
    
    # Event handlers
    def respond(message, chat_history, system_prompt, temperature, max_tokens, top_p):
        bot_response = chat(message, chat_history, system_prompt, temperature, max_tokens, top_p)
        chat_history.append((message, bot_response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot, system_prompt, temperature, max_tokens, top_p], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot, system_prompt, temperature, max_tokens, top_p], [msg, chatbot])

# ====================================================================
#  LAUNCH
# ====================================================================
if __name__ == "__main__":
    print("\n🚀 Starting AksaraLLM Chat Demo...")
    print(f"📍 Open: http://localhost:{cli_args.port}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=cli_args.port,
        share=False,
        show_error=True
    )
