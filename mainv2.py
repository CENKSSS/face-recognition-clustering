import os
import sys
import shutil
import warnings

warnings.filterwarnings("ignore")

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
RESIM_KLASORU = r"C:\Users\cenk_\Downloads\archive\lfw-deepfunneled\lfw-deepfunneled"
CIKTI_KLASORU = "Kişiler"
ESIK = 0.45
KALITE_ESIK = 0.35
MAX_YAW = 63
MIN_YUZ_ALAN_ORANI = 0.085
DESTEKLENED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from insightface.app import FaceAnalysis

def model_yukle():
    print("📦 Model yükleniyor...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Model hazır.\n")
    return app

def resimleri_topla(klasor):
    dosyalar = []
    for root, dirs, files in os.walk(klasor):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in DESTEKLENED_EXT:
                dosyalar.append(os.path.join(root, f))
    return dosyalar

def yuz_kalite_skoru(yuz_img):
    if yuz_img is None or yuz_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(yuz_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    parlaklik = gray.mean()
    h, w = yuz_img.shape[:2]
    blur_norm = min(1.0, blur / 500.0)
    parlaklik_norm = 1.0 - abs(parlaklik - 128) / 128
    boyut_norm = min(1.0, (h * w) / 10000)  # 100×100 = 10000
    return (blur_norm * 0.5) + (parlaklik_norm * 0.3) + (boyut_norm * 0.2)

def landmark_confidence(yuz):
    if not hasattr(yuz, "det_score") or yuz.det_score is None:
        return 1.0
    det = float(yuz.det_score)
    if det >= 0.85:
        return 1.0
    elif det >= 0.70:
        return 0.85
    elif det >= 0.55:
        return 0.65
    else:
        return 0.40


def poz_carpani(yuz):
    if hasattr(yuz, "pose") and yuz.pose is not None:
        yaw = abs(float(yuz.pose[1]))
        if yaw > MAX_YAW:
            return 0.6
    return 1.0

def embedding_cikar(app, dosya_listesi):
    embeddings = []
    meta = []
    cop_sayisi = 0
    alan_atlanan = 0
    toplam = len(dosya_listesi)

    for idx, dosya in enumerate(dosya_listesi, 1):
        if idx % 100 == 0:
            print(f"   İşleniyor: {idx}/{toplam} ({100 * idx / toplam:.1f}%)")

        img = cv2.imread(dosya)
        if img is None:
            continue

        yuzler = app.get(img)

        # Fallback: küçük resimleri büyüt
        if not yuzler:
            h, w = img.shape[:2]
            if max(h, w) < 800:
                buyuk = cv2.resize(img, None, fx=1.5, fy=1.5)
                yuzler = app.get(buyuk)
                if yuzler:
                    img = buyuk

        if not yuzler:
            continue

        # ── EN İYİ YÜZ SEÇİMİ (BOYUT + MERKEZİLİK) ────────────────────────────
        img_h, img_w = img.shape[:2]
        merkez_x = img_w / 2
        merkez_y = img_h / 2
        max_uzaklik = np.sqrt(merkez_x**2 + merkez_y**2)

        # En büyük yüz alanını bul (normalize için)
        max_alan = max(
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            for f in yuzler
        )

        # Her yüz için skor hesapla
        skorlar = []
        for yuz in yuzler:
            box = yuz.bbox
            alan = (box[2] - box[0]) * (box[3] - box[1])

            # Yüz merkezi
            yuz_merkez_x = (box[0] + box[2]) / 2
            yuz_merkez_y = (box[1] + box[3]) / 2

            # Merkeze uzaklık
            uzaklik = np.sqrt(
                (yuz_merkez_x - merkez_x)**2 +
                (yuz_merkez_y - merkez_y)**2
            )

            # Normalize skorlar
            boyut_skoru = alan / max_alan
            merkezilik_skoru = 1.0 - (uzaklik / max_uzaklik)

            # Final skor: %60 boyut, %40 merkezilik
            final_skor = (boyut_skoru * 0.6) + (merkezilik_skoru * 0.4)
            skorlar.append(final_skor)

        # En yüksek skorlu yüzü seç
        en_iyi_idx = int(np.argmax(skorlar))
        yuz = yuzler[en_iyi_idx]
        # ─────────────────────────────────────────────────────────────────────

        box = yuz.bbox.astype(int)
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(img.shape[1], box[2])
        y2 = min(img.shape[0], box[3])
        yuz_img = img[y1:y2, x1:x2]

        # Alan filtresi
        yuz_alani = (x2 - x1) * (y2 - y1)
        foto_alani = img.shape[0] * img.shape[1]
        alan_orani = yuz_alani / foto_alani if foto_alani > 0 else 0

        if alan_orani < MIN_YUZ_ALAN_ORANI:
            alan_atlanan += 1
            continue

        # Kalite hesaplama
        temel_kalite = yuz_kalite_skoru(yuz_img)
        lm_carpan = landmark_confidence(yuz)
        poz_carp = poz_carpani(yuz)
        kalite = temel_kalite * lm_carpan * poz_carp

        if kalite < KALITE_ESIK:
            cop_sayisi += 1
            continue

        if yuz.embedding is None:
            continue

        emb = yuz.embedding / np.linalg.norm(yuz.embedding)

        embeddings.append(emb)
        meta.append({
            "dosya": dosya,
            "yuz_idx": 0,
            "yuz_img": yuz_img,
            "kalite": kalite,
            "alan_orani": alan_orani,
        })

    print(f"\n🗑️  Düşük kalite nedeniyle atlanan: {cop_sayisi}")
    print(f"📐 Çok küçük yüz nedeniyle atlanan: {alan_atlanan}")
    return embeddings, meta


def gruplari_olustur(embeddings, meta, esik):
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [{"embeddings": embeddings, "meta": meta}]

    X = normalize(np.array(embeddings))
    cosine_sim = np.dot(X, X.T)
    dist_matrix = np.clip(1.0 - cosine_sim, 0.0, 2.0).astype(np.float32)

    db = DBSCAN(eps=esik, min_samples=1, metric="precomputed")
    etiketler = db.fit_predict(dist_matrix)

    grup_dict = {}
    for idx, etiket in enumerate(etiketler):
        if etiket not in grup_dict:
            grup_dict[etiket] = {"embeddings": [], "meta": []}
        grup_dict[etiket]["embeddings"].append(embeddings[idx])
        grup_dict[etiket]["meta"].append(meta[idx])

    return list(grup_dict.values())


def centroid_remerge(gruplar, esik):
    """Klasik O(n²) — grup sayısı < 500 için."""
    if len(gruplar) < 2:
        return gruplar

    birlestirildi = True
    while birlestirildi:
        birlestirildi = False

        centroidler = []
        for g in gruplar:
            kaliteler = np.array([m["kalite"] for m in g["meta"]])
            agirliklar = kaliteler / kaliteler.sum()
            cent = np.average(g["embeddings"], axis=0, weights=agirliklar)
            centroidler.append(cent / np.linalg.norm(cent))

        for i in range(len(gruplar)):
            for j in range(i + 1, len(gruplar)):
                mesafe = float(1.0 - np.dot(centroidler[i], centroidler[j]))
                if mesafe < esik * 1.05:
                    gruplar[i]["embeddings"] += gruplar[j]["embeddings"]
                    gruplar[i]["meta"] += gruplar[j]["meta"]
                    gruplar.pop(j)
                    birlestirildi = True
                    break
            if birlestirildi:
                break

    return gruplar


def centroid_remerge_optimized(gruplar, esik):
    """KD-Tree ile O(n log n) — grup sayısı 500-10000 arası için."""
    if len(gruplar) < 2:
        return gruplar

    print(f"   Centroid hesaplanıyor...")
    centroidler = []
    for g in gruplar:
        kaliteler = np.array([m["kalite"] for m in g["meta"]])
        agirliklar = kaliteler / kaliteler.sum()
        cent = np.average(g["embeddings"], axis=0, weights=agirliklar)
        centroidler.append(cent / np.linalg.norm(cent))

    centroidler_arr = np.array(centroidler)

    print(f"   KD-Tree ile yakın komşular bulunuyor...")
    nbrs = NearestNeighbors(n_neighbors=min(20, len(centroidler_arr)), metric="cosine")
    nbrs.fit(centroidler_arr)
    distances, indices = nbrs.kneighbors(centroidler_arr)

    print(f"   Gruplar birleştiriliyor...")
    merged = set()
    for i in range(len(gruplar)):
        if i in merged:
            continue
        for j_idx in range(1, len(indices[i])):
            j = indices[i][j_idx]
            if j in merged or j <= i:
                continue
            if distances[i][j_idx] < esik * 1.05:
                gruplar[i]["embeddings"] += gruplar[j]["embeddings"]
                gruplar[i]["meta"] += gruplar[j]["meta"]
                merged.add(j)

    final_gruplar = [g for idx, g in enumerate(gruplar) if idx not in merged]
    print(f"   {len(merged)} grup birleştirildi.")
    return final_gruplar


def klasorlere_kaydet(gruplar, cikti):
    os.makedirs(cikti, exist_ok=True)
    gruplar = sorted(gruplar, key=lambda g: -len(g["meta"]))

    print(f"\n📁 Çıktı: {os.path.abspath(cikti)}\n")

    for sira, grup in enumerate(gruplar, start=1):
        klasor = os.path.join(cikti, f"kisi{sira}")
        os.makedirs(klasor, exist_ok=True)

        for m in grup["meta"]:
            taban = os.path.splitext(os.path.basename(m["dosya"]))[0]
            if m["yuz_img"].size > 0:
                cv2.imwrite(
                    os.path.join(klasor, f"{taban}_yuz{m['yuz_idx']}.jpg"),
                    m["yuz_img"],
                )
            shutil.copy2(
                m["dosya"],
                os.path.join(klasor, f"org_{taban}_yuz{m['yuz_idx']}.jpg"),
            )

        if sira <= 20:
            ort_kalite = np.mean([m["kalite"] for m in grup["meta"]])
            print(
                f"   👤 kisi{sira}  →  {len(grup['meta'])} fotoğraf "
                f"| ort. kalite: {ort_kalite:.2f}"
            )

    if len(gruplar) > 20:
        print(f"   ... ve {len(gruplar) - 20} grup daha")

    return len(gruplar)


def main():
    if not os.path.isdir(RESIM_KLASORU):
        print(f"❌ '{RESIM_KLASORU}' klasörü bulunamadı!")
        sys.exit(1)

    dosya_listesi = resimleri_topla(RESIM_KLASORU)
    if not dosya_listesi:
        print(f"❌ '{RESIM_KLASORU}' içinde resim yok.")
        sys.exit(1)

    print(f"🖼️  {len(dosya_listesi)} resim bulundu\n")

    app = model_yukle()

    print("🔍 Yüzler tespit ediliyor...\n")
    embeddings, meta = embedding_cikar(app, dosya_listesi)

    if not embeddings:
        print("❌ Hiç yüz bulunamadı.")
        sys.exit(1)

    print(f"\n📊 {len(embeddings)} yüz vektörü elde edildi.")
    print(f" Stage 1 — DBSCAN (eşik: {ESIK})...")
    gruplar = gruplari_olustur(embeddings, meta, ESIK)
    print(f"   → {len(gruplar)} grup oluştu.")

    grup_sayisi = len(gruplar)
    if grup_sayisi < 500:
        print(f"\n🔗 Stage 2 — Centroid re-merge (klasik)...")
        gruplar = centroid_remerge(gruplar, ESIK)
    elif grup_sayisi < 10000:
        print(f"\n🔗 Stage 2 — Centroid re-merge (KD-Tree optimize)...")
        gruplar = centroid_remerge_optimized(gruplar, ESIK)
    else:
        print(f"\n⏭️  Stage 2 atlandı (grup sayısı çok yüksek: {grup_sayisi})")

    print(f"   → {len(gruplar)} final grup.\n")

    kisi_sayisi = klasorlere_kaydet(gruplar, CIKTI_KLASORU)

    print(f"\n✅ Tamamlandı! {kisi_sayisi} farklı kişi tespit edildi.")
    print(f"   Klasör: {os.path.abspath(CIKTI_KLASORU)}")


if __name__ == "__main__":
    main()
