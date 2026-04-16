#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║  🇮🇩 AksaraLLM CLI v1.0                              ║
║  Interactive AI Assistant — Berbahasa Indonesia      ║
║  Open Source | Apache 2.0 | github.com/AksaraLLM    ║
╚══════════════════════════════════════════════════════╝

Usage:
  python aksara_cli.py                    # Interactive mode
  python aksara_cli.py --model local      # Use local 26M model
  python aksara_cli.py --model qwen       # Use Qwen via HF API
  python aksara_cli.py --model ollama     # Use Ollama local
"""

import os
import sys
import json
import argparse
import subprocess
import datetime
import glob
from pathlib import Path

# ═══════════════════════════════════════
# STYLING — Rich terminal UI
# ═══════════════════════════════════════

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.tree import Tree
    from rich.prompt import Prompt
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class AksaraCLI:
    """AksaraLLM CLI — Your Indonesian AI Assistant in the Terminal"""

    VERSION = "1.0.0"
    
    SYSTEM_PROMPT = """Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia.

Aturan:
1. Jawab dalam bahasa Indonesia kecuali diminta lain
2. Jujur — jika tidak tahu, bilang tidak tahu
3. Berikan jawaban informatif dan terstruktur
4. Jika diminta baca file, gunakan tools yang tersedia
5. Hormati budaya dan nilai Indonesia

Identitas:
- Nama: AksaraLLM
- Pembuat: Komunitas Open-Source Indonesia
- Lisensi: Apache 2.0
"""

    COMMANDS = {
        "/help": "Tampilkan bantuan",
        "/file <path>": "Baca isi file",
        "/ls <dir>": "List isi direktori",
        "/tree <dir>": "Tampilkan tree direktori",
        "/search <query>": "Cari file berdasarkan nama",
        "/run <cmd>": "Jalankan perintah shell",
        "/edit <path>": "Edit file (buka di editor)",
        "/history": "Tampilkan riwayat chat",
        "/clear": "Bersihkan layar",
        "/save": "Simpan percakapan",
        "/model": "Info model yang digunakan",
        "/quit": "Keluar",
    }

    # Safety: Blocked commands
    BLOCKED_COMMANDS = [
        "rm -rf /", "rm -rf ~", "rm -rf *", "mkfs", "dd if=",
        "> /dev/sda", "chmod -R 777 /", "wget http", "curl http",
        ":(){ :|:& };:",  # fork bomb
        "shutdown", "reboot", "poweroff", "init 0",
        "passwd", "useradd", "userdel", "groupadd",
        "iptables", "ufw", "systemctl",
        "npm publish", "pip install --",
    ]
    
    # Safety: Sensitive files that cannot be read
    SENSITIVE_FILES = [
        ".env", ".ssh/", "id_rsa", "id_ed25519", ".aws/",
        "credentials", "password", "secret", "token",
        ".gnupg/", ".netrc", ".pgpass",
    ]

    def __init__(self, model_type="qwen", model_path=None, safe_mode=True):
        self.console = Console() if HAS_RICH else None
        self.model_type = model_type
        self.model_path = model_path
        self.safe_mode = safe_mode
        self.history = []
        self.model = None
        self.tokenizer = None
        self.session_start = datetime.datetime.now()
        self.cwd = os.getcwd()
        
    # ═══════════════════════════════════
    # UI METHODS
    # ═══════════════════════════════════
    
    def print_banner(self):
        banner = """
[bold cyan]╔══════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]  [bold white]🇮🇩 AksaraLLM CLI[/bold white] [dim]v{version}[/dim]                            [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]  [dim]Interactive AI Assistant — Berbahasa Indonesia[/dim]      [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]  [dim]Ketik /help untuk bantuan | /quit untuk keluar[/dim]     [bold cyan]║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════════════╝[/bold cyan]
""".format(version=self.VERSION)
        
        if self.console:
            self.console.print(banner)
        else:
            print(f"\n🇮🇩 AksaraLLM CLI v{self.VERSION}")
            print("Ketik /help untuk bantuan | /quit untuk keluar\n")

    def print_help(self):
        if self.console:
            table = Table(title="📖 Perintah AksaraLLM CLI", 
                         border_style="cyan", show_header=True)
            table.add_column("Perintah", style="bold green")
            table.add_column("Fungsi", style="white")
            for cmd, desc in self.COMMANDS.items():
                table.add_row(cmd, desc)
            self.console.print(table)
        else:
            print("\n📖 Perintah:")
            for cmd, desc in self.COMMANDS.items():
                print(f"  {cmd:20s} — {desc}")
            print()

    def print_response(self, text):
        if self.console:
            self.console.print(Panel(
                Markdown(text),
                title="[bold green]🤖 AksaraLLM[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
        else:
            print(f"\n🤖 AksaraLLM:\n{text}\n")

    def print_error(self, text):
        if self.console:
            self.console.print(f"[bold red]❌ Error:[/bold red] {text}")
        else:
            print(f"❌ Error: {text}")

    def print_info(self, text):
        if self.console:
            self.console.print(f"[bold blue]ℹ️[/bold blue] {text}")
        else:
            print(f"ℹ️ {text}")

    def print_success(self, text):
        if self.console:
            self.console.print(f"[bold green]✅[/bold green] {text}")
        else:
            print(f"✅ {text}")

    # ═══════════════════════════════════
    # MODEL BACKENDS
    # ═══════════════════════════════════

    def load_model(self):
        """Load AI model based on selected backend"""
        if self.model_type == "local":
            return self._load_local_model()
        elif self.model_type == "qwen":
            return self._load_qwen_api()
        elif self.model_type == "ollama":
            return self._load_ollama()
        else:
            self.print_info("Mode tanpa model — hanya tools yang aktif")
            return True

    def _load_local_model(self):
        """Load AksaraLLM 26M locally"""
        try:
            import torch
            from tokenizers import ByteLevelBPETokenizer
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from aksarallm.model import aksaraLLMModel
            from aksarallm.config import aksaraLLMConfig

            # Find model files
            model_path = self.model_path or self._find_model()
            if not model_path:
                self.print_error("Model tidak ditemukan. Gunakan --model-path")
                return False

            # Find tokenizer  
            tok_dir = self._find_tokenizer()
            if not tok_dir:
                self.print_error("Tokenizer tidak ditemukan")
                return False

            self.print_info(f"Loading model: {model_path}")
            self.tokenizer = ByteLevelBPETokenizer(
                f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            cfg = aksaraLLMConfig(**{k: v for k, v in ckpt["config"].items()
                                    if k in aksaraLLMConfig.__dataclass_fields__})
            self.model = aksaraLLMModel(cfg).to(device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.model.eval()
            self.device = device
            
            params = sum(p.numel() for p in self.model.parameters()) / 1e6
            self.print_success(f"AksaraLLM {params:.1f}M loaded on {device}")
            return True
        except Exception as e:
            self.print_error(f"Gagal load model: {e}")
            return False

    def _load_qwen_api(self):
        """Use Qwen via HuggingFace Inference API"""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient("Qwen/Qwen2.5-0.5B-Instruct")
            self.print_success("Qwen2.5 via HuggingFace API ready")
            return True
        except ImportError:
            self.print_info("Fallback: menggunakan requests langsung")
            self.client = None
            return True

    def _load_ollama(self):
        """Use Ollama for local inference"""
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                self.print_success("Ollama connected!")
                return True
        except:
            self.print_error("Ollama tidak berjalan. Jalankan: ollama serve")
            return False

    def _find_model(self):
        """Auto-find model checkpoint"""
        search_paths = [
            os.path.expanduser("~/aksaraLLM-data/sft_v4_distill_best.pt"),
            os.path.expanduser("~/aksaraLLM-data/sft_v3_facts.pt"),
            os.path.expanduser("~/aksaraLLM-data/sft_v3_best.pt"),
            "./checkpoints/sft_v3_facts.pt",
            "./model/checkpoints/sft_v3_facts.pt",
        ]
        for p in search_paths:
            if os.path.exists(p):
                return p
        return None

    def _find_tokenizer(self):
        """Auto-find tokenizer directory"""
        search_paths = [
            os.path.expanduser("~/aksara-tokenizer-id"),
            "./tokenizer",
            "./model/tokenizer",
        ]
        for p in search_paths:
            if os.path.exists(os.path.join(p, "vocab.json")):
                return p
        return None

    # ═══════════════════════════════════
    # GENERATION
    # ═══════════════════════════════════

    def generate(self, user_input, context=""):
        """Generate response from AI"""
        
        # Add file context if available
        full_prompt = user_input
        if context:
            full_prompt = f"Konteks file:\n```\n{context}\n```\n\nPertanyaan: {user_input}"

        if self.model_type == "local":
            return self._generate_local(full_prompt)
        elif self.model_type == "qwen":
            return self._generate_qwen(full_prompt)
        elif self.model_type == "ollama":
            return self._generate_ollama(full_prompt)
        else:
            return "⚠️ Tidak ada model aktif. Gunakan --model untuk memilih backend."

    def _generate_local(self, prompt):
        """Generate with local AksaraLLM 26M"""
        import torch
        
        text = f"### Instruksi:\n{prompt}\n\n### Respons:\n"
        ids = torch.tensor([self.tokenizer.encode(text).ids[-150:]], device=self.device)
        bad = {self.tokenizer.token_to_id(t) for t in ["<pad>", "<s>", "<unk>"] 
               if self.tokenizer.token_to_id(t) is not None}
        eos = self.tokenizer.token_to_id("</s>")
        
        with torch.no_grad():
            for _ in range(150):
                logits, _ = self.model(ids[:, -256:])
                logits = logits[0, -1, :] / 0.7
                for b in bad:
                    logits[b] = -float("inf")
                for t in set(ids[0, -50:].tolist()):
                    logits[t] /= 1.3
                nxt = torch.multinomial(torch.softmax(logits, -1), 1)
                ids = torch.cat([ids, nxt.view(1, 1)], 1)
                if eos and nxt.item() == eos:
                    break
        
        out = self.tokenizer.decode(ids[0].tolist()).split("### Respons:")[-1]
        for tag in ["<pad>", "<s>", "</s>", "<unk>"]:
            out = out.replace(tag, "")
        return out.strip()

    def _generate_qwen(self, prompt):
        """Generate with Qwen via HuggingFace API"""
        try:
            if self.client:
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            else:
                # Fallback to requests
                import requests
                r = requests.post(
                    "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct",
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
                    timeout=30
                )
                return r.json()[0]["generated_text"]
        except Exception as e:
            return f"⚠️ API error: {e}"

    def _generate_ollama(self, prompt):
        """Generate with Ollama"""
        try:
            import requests
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": "qwen2.5:0.5b",
                "prompt": prompt,
                "system": self.SYSTEM_PROMPT,
                "stream": False,
            }, timeout=60)
            return r.json()["response"]
        except Exception as e:
            return f"⚠️ Ollama error: {e}"

    # ═══════════════════════════════════
    # TOOLS — File Operations
    # ═══════════════════════════════════

    def _is_sensitive_file(self, path):
        """Check if file is sensitive"""
        path_lower = path.lower()
        return any(s in path_lower for s in self.SENSITIVE_FILES)

    def _is_command_safe(self, cmd):
        """Check if command is safe to run"""
        cmd_lower = cmd.lower().strip()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return False, blocked
        return True, ""

    def cmd_file(self, path):
        """Read and display file contents"""
        path = os.path.expanduser(path.strip())
        if not os.path.exists(path):
            self.print_error(f"File tidak ditemukan: {path}")
            return None
        
        # Safety check
        if self.safe_mode and self._is_sensitive_file(path):
            self.print_error(f"🔒 File sensitif — akses ditolak: {path}")
            self.print_info("Gunakan --no-safe-mode jika yakin")
            return None
        
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Limit file size display
            if len(content) > 50000:
                self.print_info(f"File besar ({len(content)} chars) — menampilkan 50,000 pertama")
                content = content[:50000]
            
            if self.console:
                # Detect language for syntax highlighting
                ext = os.path.splitext(path)[1]
                lang_map = {
                    ".py": "python", ".js": "javascript", ".ts": "typescript",
                    ".html": "html", ".css": "css", ".json": "json",
                    ".sh": "bash", ".yml": "yaml", ".yaml": "yaml",
                    ".md": "markdown", ".sql": "sql", ".java": "java",
                    ".cpp": "cpp", ".c": "c", ".go": "go", ".rs": "rust",
                }
                lang = lang_map.get(ext, "text")
                
                syntax = Syntax(content, lang, theme="monokai", 
                              line_numbers=True, word_wrap=True)
                self.console.print(Panel(syntax, title=f"📄 {path}", 
                                        border_style="blue"))
            else:
                print(f"\n📄 {path}:")
                print(content)
            
            return content
        except Exception as e:
            self.print_error(f"Gagal baca file: {e}")
            return None

    def cmd_ls(self, path="."):
        """List directory contents"""
        path = os.path.expanduser(path.strip() or ".")
        if not os.path.isdir(path):
            self.print_error(f"Bukan direktori: {path}")
            return
        
        items = sorted(os.listdir(path))
        
        if self.console:
            table = Table(title=f"📁 {os.path.abspath(path)}", border_style="blue")
            table.add_column("Nama", style="white")
            table.add_column("Tipe", style="cyan")
            table.add_column("Ukuran", style="green", justify="right")
            
            for item in items:
                full = os.path.join(path, item)
                if os.path.isdir(full):
                    children = len(os.listdir(full)) if os.access(full, os.R_OK) else "?"
                    table.add_row(f"📁 {item}", "DIR", f"{children} items")
                else:
                    size = os.path.getsize(full)
                    if size > 1024*1024:
                        size_str = f"{size/1024/1024:.1f} MB"
                    elif size > 1024:
                        size_str = f"{size/1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    table.add_row(f"📄 {item}", os.path.splitext(item)[1] or "file", size_str)
            
            self.console.print(table)
        else:
            print(f"\n📁 {os.path.abspath(path)}:")
            for item in items:
                full = os.path.join(path, item)
                prefix = "📁" if os.path.isdir(full) else "📄"
                print(f"  {prefix} {item}")

    def cmd_tree(self, path=".", max_depth=3, _depth=0):
        """Display directory tree"""
        path = os.path.expanduser(path.strip() or ".")
        
        if self.console and _depth == 0:
            tree = Tree(f"📁 {os.path.abspath(path)}")
            self._build_tree(tree, path, max_depth, 0)
            self.console.print(tree)
        elif _depth == 0:
            print(f"📁 {os.path.abspath(path)}")
            self._print_tree(path, max_depth, 0, "")

    def _build_tree(self, tree, path, max_depth, depth):
        if depth >= max_depth:
            return
        try:
            for item in sorted(os.listdir(path)):
                if item.startswith("."):
                    continue
                full = os.path.join(path, item)
                if os.path.isdir(full):
                    branch = tree.add(f"📁 {item}")
                    self._build_tree(branch, full, max_depth, depth + 1)
                else:
                    tree.add(f"📄 {item}")
        except PermissionError:
            tree.add("[red]🔒 Permission denied[/red]")

    def _print_tree(self, path, max_depth, depth, prefix):
        if depth >= max_depth:
            return
        try:
            items = sorted(os.listdir(path))
            for i, item in enumerate(items):
                if item.startswith("."):
                    continue
                full = os.path.join(path, item)
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                icon = "📁" if os.path.isdir(full) else "📄"
                print(f"{prefix}{connector}{icon} {item}")
                if os.path.isdir(full):
                    ext = "    " if is_last else "│   "
                    self._print_tree(full, max_depth, depth + 1, prefix + ext)
        except PermissionError:
            pass

    def cmd_search(self, query):
        """Search for files matching query"""
        query = query.strip()
        results = []
        for root, dirs, files in os.walk(self.cwd):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if query.lower() in f.lower():
                    results.append(os.path.join(root, f))
            if len(results) >= 50:
                break
        
        if results:
            if self.console:
                table = Table(title=f"🔍 Hasil pencarian: '{query}'", border_style="yellow")
                table.add_column("File", style="white")
                table.add_column("Path", style="dim")
                for r in results[:20]:
                    table.add_row(os.path.basename(r), os.path.dirname(r))
                self.console.print(table)
            else:
                print(f"\n🔍 Hasil untuk '{query}':")
                for r in results[:20]:
                    print(f"  📄 {r}")
        else:
            self.print_info(f"Tidak ada file yang cocok dengan '{query}'")

    def cmd_run(self, cmd):
        """Execute shell command with safety checks"""
        cmd = cmd.strip()
        
        # Safety: check blocked commands
        if self.safe_mode:
            is_safe, blocked = self._is_command_safe(cmd)
            if not is_safe:
                self.print_error(f"🚫 Perintah DIBLOKIR — mengandung: '{blocked}'")
                self.print_info("Perintah ini berpotensi berbahaya")
                return None
        
        # Confirmation prompt
        self.print_info(f"Perintah: {cmd}")
        confirm = input("⚠️  Lanjutkan? (y/N): ").strip().lower()
        if confirm != "y":
            self.print_info("Dibatalkan.")
            return None
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                   text=True, timeout=30)
            if result.stdout:
                if self.console:
                    self.console.print(Panel(result.stdout, title="Output", 
                                           border_style="green"))
                else:
                    print(result.stdout)
            if result.stderr:
                if self.console:
                    self.console.print(f"[yellow]{result.stderr}[/yellow]")
                else:
                    print(f"⚠️ {result.stderr}")
            return result.stdout
        except subprocess.TimeoutExpired:
            self.print_error("Command timeout (30s)")
            return None
        except Exception as e:
            self.print_error(f"Gagal: {e}")
            return None

    def cmd_edit(self, path):
        """Open file in default editor"""
        path = path.strip()
        editor = os.environ.get("EDITOR", "nano")
        os.system(f"{editor} {path}")

    def cmd_history(self):
        """Show chat history"""
        if not self.history:
            self.print_info("Belum ada riwayat chat")
            return
        
        if self.console:
            for i, (role, msg) in enumerate(self.history):
                icon = "👤" if role == "user" else "🤖"
                color = "blue" if role == "user" else "green"
                self.console.print(f"[{color}]{icon} [{i+1}] {msg[:100]}...[/{color}]"
                                  if len(msg) > 100 else f"[{color}]{icon} [{i+1}] {msg}[/{color}]")
        else:
            for i, (role, msg) in enumerate(self.history):
                icon = "👤" if role == "user" else "🤖"
                print(f"{icon} [{i+1}] {msg[:100]}")

    def cmd_save(self):
        """Save conversation to file"""
        filename = f"aksara_chat_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "session": self.session_start.isoformat(),
                "model": self.model_type,
                "history": [{"role": r, "message": m} for r, m in self.history]
            }, f, ensure_ascii=False, indent=2)
        self.print_success(f"Percakapan disimpan ke {filename}")

    def cmd_model_info(self):
        """Show current model info"""
        if self.console:
            table = Table(title="🤖 Model Info", border_style="cyan")
            table.add_column("Property", style="bold")
            table.add_column("Value")
            table.add_row("Backend", self.model_type)
            table.add_row("Session", self.session_start.strftime("%Y-%m-%d %H:%M"))
            table.add_row("Messages", str(len(self.history)))
            table.add_row("CWD", self.cwd)
            self.console.print(table)
        else:
            print(f"\n🤖 Backend: {self.model_type}")
            print(f"📊 Messages: {len(self.history)}")

    # ═══════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════

    def process_input(self, user_input):
        """Process user input — command or chat"""
        user_input = user_input.strip()
        if not user_input:
            return

        # Commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                self.print_help()
            elif cmd == "/file":
                content = self.cmd_file(arg)
                if content and arg:
                    # Ask AI about the file
                    self.print_info("File dimuat. Ketik pertanyaan tentang file ini.")
                    self.history.append(("system", f"[File loaded: {arg}]"))
            elif cmd == "/ls":
                self.cmd_ls(arg)
            elif cmd == "/tree":
                self.cmd_tree(arg)
            elif cmd == "/search":
                self.cmd_search(arg)
            elif cmd == "/run":
                self.cmd_run(arg)
            elif cmd == "/edit":
                self.cmd_edit(arg)
            elif cmd == "/history":
                self.cmd_history()
            elif cmd == "/save":
                self.cmd_save()
            elif cmd == "/model":
                self.cmd_model_info()
            elif cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                self.print_banner()
            elif cmd in ["/quit", "/exit", "/q"]:
                self.print_info("Sampai jumpa! 👋")
                return "QUIT"
            else:
                self.print_error(f"Perintah tidak dikenal: {cmd}. Ketik /help")
            return

        # Chat with AI
        self.history.append(("user", user_input))
        
        # Check if there's recent file context
        context = ""
        for role, msg in reversed(self.history[-5:]):
            if role == "system" and "File loaded:" in msg:
                filepath = msg.split("File loaded: ")[1].rstrip("]")
                try:
                    with open(filepath, "r") as f:
                        context = f.read()[:2000]
                except:
                    pass
                break

        if self.console:
            with self.console.status("[bold green]🤔 AksaraLLM sedang berpikir...[/bold green]"):
                response = self.generate(user_input, context)
        else:
            print("🤔 Berpikir...")
            response = self.generate(user_input, context)
        
        self.history.append(("assistant", response))
        self.print_response(response)

    def run(self):
        """Main interactive loop"""
        self.print_banner()
        
        # Load model
        if not self.load_model():
            self.print_info("Melanjutkan tanpa model AI (hanya tools)")
            self.model_type = "none"
        
        print()
        
        while True:
            try:
                if self.console:
                    user_input = Prompt.ask("[bold blue]👤 Kamu[/bold blue]")
                else:
                    user_input = input("👤 > ")
                
                result = self.process_input(user_input)
                if result == "QUIT":
                    break
                    
            except KeyboardInterrupt:
                print()
                self.print_info("Ctrl+C — Ketik /quit untuk keluar")
            except EOFError:
                break


# ═══════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🇮🇩 AksaraLLM CLI — AI Assistant Berbahasa Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python aksara_cli.py                      # Default (Qwen API)
  python aksara_cli.py --model local        # AksaraLLM 26M lokal
  python aksara_cli.py --model ollama       # Ollama backend
  python aksara_cli.py --model none         # Hanya tools, tanpa AI
        """
    )
    parser.add_argument("--model", choices=["local", "qwen", "ollama", "none"],
                       default="qwen", help="Backend model (default: qwen)")
    parser.add_argument("--model-path", help="Path ke checkpoint model (.pt)")
    parser.add_argument("--no-safe-mode", action="store_true",
                       help="Matikan safety guard (HATI-HATI!)")
    
    args = parser.parse_args()
    
    cli = AksaraCLI(model_type=args.model, model_path=args.model_path,
                    safe_mode=not args.no_safe_mode)
    cli.run()


if __name__ == "__main__":
    main()
