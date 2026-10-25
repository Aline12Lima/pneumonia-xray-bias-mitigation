# Medição de tempo — F2 passo 3: 1 época de DenseNet121 a 320 px com 1.000 imagens
# Usa imagens sintéticas: o tempo de treino não depende do conteúdo; nenhum dado sai do Redivis.
import time, json, os, torch, torchvision

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(os.cpu_count())

N, LOTE, PX = 1000, 32, 320   # lote 32 = receita do protocolo (§7.2); a medição em CPU de 24/10 usou 16
modelo = torchvision.models.densenet121(weights="IMAGENET1K_V1")
modelo.classifier = torch.nn.Linear(modelo.classifier.in_features, 1)
modelo = modelo.to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
perda = torch.nn.BCEWithLogitsLoss()
usar_amp = dev == "cuda"
scaler = torch.amp.GradScaler(enabled=usar_amp)

def lote():
    x = torch.randn(LOTE, 3, PX, PX, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, 2, (LOTE, 1), device=dev).float()
    return x, y

def sinc():
    if dev == "cuda":
        torch.cuda.synchronize()

# aquecimento (não entra na conta)
modelo.train()
for _ in range(2):
    x, y = lote()
    with torch.autocast(dev, enabled=usar_amp):
        l = perda(modelo(x), y)
    opt.zero_grad(); scaler.scale(l).backward(); scaler.step(opt); scaler.update()
sinc()

t0 = time.time()
for _ in range(N // LOTE + (N % LOTE > 0)):
    x, y = lote()
    with torch.autocast(dev, enabled=usar_amp):
        l = perda(modelo(x), y)
    opt.zero_grad(); scaler.scale(l).backward(); scaler.step(opt); scaler.update()
sinc()
t_treino = time.time() - t0

modelo.eval(); t0 = time.time()
with torch.no_grad(), torch.autocast(dev, enabled=usar_amp):
    for _ in range(N // LOTE + (N % LOTE > 0)):
        modelo(lote()[0])
sinc()
t_inf = time.time() - t0

r = {"dispositivo": dev,
     "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "sem GPU",
     "threads_cpu": torch.get_num_threads(), "imagens": N, "lote": LOTE, "px": PX, "amp": usar_amp,
     "treino_1000_img_s": round(t_treino, 1), "treino_img_por_s": round(N / t_treino, 1),
     "inferencia_1000_img_s": round(t_inf, 1), "inferencia_img_por_s": round(N / t_inf, 1)}
print(json.dumps(r, indent=2, ensure_ascii=False))
json.dump(r, open("tempo_epoca.json", "w"), indent=2, ensure_ascii=False)
