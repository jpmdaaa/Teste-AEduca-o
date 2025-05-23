import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from main import Sistema
import threading
import time

class InterfaceTk:
    def __init__(self, root):
        self.root = root
        self.root.title("+A Educação - Tutor Adaptativo")
        
        self.sistema = None
        self.historico = []
        self.processando = False
        self.formato = "texto"
        self.nivel = "iniciante"
        self.sistema_config = {
            "ollama_model": "llama2",
            "whisper_model": "small",
            "pasta_dados": "dados",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "modo_quieto": True
        }
        
      
        self._construir_interface()
        self._inicializar_sistema()
        
    def _inicializar_sistema(self):
        try:
            self._atualizar_status("Inicializando sistema tutor adaptativo...")
            self.sistema = Sistema(config=self.sistema_config)
            documentos = self.sistema.processar_dados()
            if documentos:
                self.sistema.indexador.criar_indice(documentos)
            self._atualizar_status("✅ Sistema pronto para uso!")
        except Exception as e:
            self._atualizar_status(f"❌ Falha ao iniciar sistema: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao iniciar sistema: {str(e)}")
            
    def _construir_interface(self):
        # Configurações
        config_frame = ttk.LabelFrame(self.root, text="Configurações")
        config_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(config_frame, text="Formato:").pack(side="left")
        self.formato_var = tk.StringVar(value=self.formato)
        formato_menu = ttk.Combobox(config_frame, textvariable=self.formato_var, state="readonly")
        formato_menu['values'] = ["texto", "vídeo", "áudio"]
        formato_menu.pack(side="left", padx=5)
        
        ttk.Label(config_frame, text="Nível:").pack(side="left")
        self.nivel_var = tk.StringVar(value=self.nivel)
        nivel_menu = ttk.Combobox(config_frame, textvariable=self.nivel_var, state="readonly")
        nivel_menu['values'] = ["iniciante", "intermediário", "avançado"]
        nivel_menu.pack(side="left", padx=5)
        
        reiniciar_btn = ttk.Button(config_frame, text="Reiniciar Sistema", command=self._reiniciar_sistema)
        reiniciar_btn.pack(side="right", padx=5)

        # Área de histórico
        historico_frame = ttk.LabelFrame(self.root, text="Histórico de Conversa")
        historico_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.historico_text = scrolledtext.ScrolledText(historico_frame, wrap=tk.WORD, state='disabled', height=20)
        self.historico_text.pack(fill="both", expand=True)

        # Entrada de pergunta
        entrada_frame = ttk.Frame(self.root)
        entrada_frame.pack(fill="x", padx=10, pady=5)
        
        self.pergunta_var = tk.StringVar()
        pergunta_entry = ttk.Entry(entrada_frame, textvariable=self.pergunta_var)
        pergunta_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        enviar_btn = ttk.Button(entrada_frame, text="Enviar", command=self._enviar_pergunta)
        enviar_btn.pack(side="right")

        # Status
        self.status_var = tk.StringVar(value="✅ Sistema operacional")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(fill="x", padx=10, pady=5)
        
    def _atualizar_status(self, mensagem):
        self.status_var.set(mensagem)
        self.root.update_idletasks()

    def _atualizar_historico(self, role, content, metadata=None):
        self.historico.append({"role": role, "content": content, "metadata": metadata})
        self.historico_text.config(state='normal')
        self.historico_text.insert(tk.END, f"{role.capitalize()}: {content}\n")
        if metadata:
            self.historico_text.insert(tk.END, f"  Detalhes: {metadata}\n")
        self.historico_text.see(tk.END)
        self.historico_text.config(state='disabled')

    def _enviar_pergunta(self):
        if self.processando:
            return
        
        pergunta = self.pergunta_var.get().strip()
        if not pergunta:
            return
        
        self.pergunta_var.set("")
        threading.Thread(target=self._processar_resposta, args=(pergunta,)).start()

    def _processar_resposta(self, pergunta):
        self.processando = True
        self._atualizar_historico("Usuário", pergunta)

        resposta_completa = ""
        try:
            for chunk in self._gerar_resposta(pergunta):
                resposta_completa += chunk
                self._atualizar_resposta_parcial(resposta_completa + "▌")
                time.sleep(0.02)
                
            self._atualizar_resposta_parcial(resposta_completa)
            self._atualizar_historico("Assistente", resposta_completa, {
                "formato": self.formato_var.get(),
                "nivel": self.nivel_var.get()
            })
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar pergunta: {str(e)}")
        finally:
            self.processando = False

    def _atualizar_resposta_parcial(self, resposta):
        self.historico_text.config(state='normal')
        self.historico_text.insert(tk.END, f"Assistente: {resposta}\n")
        self.historico_text.see(tk.END)
        self.historico_text.config(state='disabled')

    def _gerar_resposta(self, pergunta):
        if not self.sistema or not hasattr(self.sistema, 'tutor'):
            yield "Sistema não está pronto para responder."
            return
        
        try:
            resposta = self.sistema.tutor.responder(
                pergunta=pergunta,
                formato=self.formato_var.get()
            )
            for i in range(0, len(resposta), 10):
                yield resposta[i:i+10]
        except Exception as e:
            yield f"Erro ao gerar resposta: {str(e)}"

    def _reiniciar_sistema(self):
        self._inicializar_sistema()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceTk(root)
    root.mainloop()
