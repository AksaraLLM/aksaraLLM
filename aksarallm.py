"""
🇮🇩 AksaraLLM — Simple Python API

Usage:
    from aksarallm import AksaraLLM
    llm = AksaraLLM()
    print(llm.chat("Apa itu Pancasila?"))
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class AksaraLLM:
    def __init__(self, model_name="AksaraLLM/aksarallm-1.5b-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()
        self.system = "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu."
        self.history = []

    def chat(self, message, temperature=0.7, max_tokens=512):
        messages = [{"role": "system", "content": self.system}]
        for u, b in self.history:
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": b}]
        messages.append({"role": "user", "content": message})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                      top_p=0.9, do_sample=True, repetition_penalty=1.15,
                                      pad_token_id=self.tokenizer.eos_token_id)
        resp = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        self.history.append((message, resp))
        return resp

    def reset(self):
        self.history = []

if __name__ == "__main__":
    print("🇮🇩 AksaraLLM Chat — ketik 'quit' untuk keluar\n")
    llm = AksaraLLM()
    while True:
        q = input("👤 Kamu: ").strip()
        if q.lower() in ("quit", "exit", "q"): break
        print(f"🤖 AksaraLLM: {llm.chat(q)}\n")
