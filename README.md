# Akıllı Görsel Arama Motoru 

Bu proje, bir sorgu görseli kullanarak devasa bir resim koleksiyonu içinde anlamsal olarak benzer görselleri bulan,
Python tabanlı, uçtan uca bir görsel arama motorudur.
Proje, basit piksel karşılaştırmasının ötesine geçerek, Derin Öğrenme (Deep Learning) ve Vektör Arama (Vector Search) tekniklerini kullanır.


---

## Teknik Ayrıntılar: Nasıl Çalışır?

Bu sistem, iki ana aşamadan oluşan modern bir arama mimarisi üzerine kuruludur:

Offline İndeksleme (`indexer.py`):
   - Öznitelik Çıkarımı (Feature Extraction): Veri setindeki her bir görsel, önceden eğitilmiş bir Evrişimli Sinir Ağı'na (`torchvision.models.resnet50`) girdi olarak verilir.
   - Vektör Temsili (Embeddings): Modelin son sınıflandırma katmanı atılarak, her görselin içeriğini temsil eden yüksek boyutlu bir özellik vektörü ($v \in \mathbb{R}^{2048}$) elde edilir. Bu vektör, görselin "parmak izi" gibidir.
   - Vektör Veritabanı: Tüm görsellere ait bu vektörler ve karşılık gelen dosya yolları, hızlı erişim için bir `features.pkl` dosyasına (veya daha büyük sistemler için Faiss gibi bir vektör indeksine) kaydedilir.

Real-Time Arama (Reflex Web App):
   - Kullanıcı bir görsel yüklediğinde, aynı öznitelik çıkarımı işlemi bu yeni görsele de uygulanır ve bir "sorgu vektörü" ($v_q$) oluşturulur.
   - Kosinüs Benzerliği (Cosine Similarity):** Sorgu vektörü, indeksteki diğer tüm vektörlerle karşılaştırılır. Bu karşılaştırma, vektörlerin büyüklüğünü değil, aralarındaki açıyı ölçen Kosinüs Benzerliği metriği ile yapılır. Bu, anlamsal yakınlığı ölçmek için idealdir.
     $$ \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$
   - Sonuç olarak, kosinüs skoru 1'e en yakın olan görseller, en benzer sonuçlar olarak kullanıcıya sunulur.

## Kullanılan Teknolojiler

- **Backend & Frontend:** [Python](https.python.org) & [Reflex](https://reflex.dev/)
- **AI & Computer Vision:** [PyTorch](https://pytorch.org/) (ResNet50 modeli ile)
- **Görsel İşleme:** [Pillow (PIL)](https://python-pillow.org/)
- **Vektör Matematiği:** [Scikit-learn](https://scikit-learn.org/) & [NumPy](https://numpy.org/)

## 📂 Proje Yapısı

<img width="317" height="424" alt="image" src="https://github.com/user-attachments/assets/eebfdc1f-833e-42ca-a2fe-33a5b686e31d" />


## ⚙️ Kurulum ve Çalıştırma

Bu projeyi yerel makinenizde çalıştırmak için iki ana adım bulunmaktadır:

Adım 1: Görselleri İndeksleme

1.  Depoyu Klonlayın ve Klasöre Girin:
    ```bash
    git clone [https://github.com/SENIN-KULLANICI-ADIN/akilli-gorsel-arama.git](https://github.com/SENIN-KULLANICI-ADIN/akilli-gorsel-arama.git)
    cd akilli-gorsel-arama
    ```
2.  Gerekli Ortamı ve Paketleri Kurun:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows için: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  Veri Setini Hazırlayın: `dataset` klasörünün içine arama yapmak istediğiniz görselleri (JPG, PNG) ekleyin.
4.  İndeksleyiciyi Çalıştırın:
    ```bash
    python indexer.py
    ```
    Bu komut, `dataset` klasöründeki tüm görselleri işleyecek ve `features.pkl` dosyasını oluşturacaktır. Bu işlem, görsel sayısına ve bilgisayarınızın gücüne bağlı olarak zaman alabilir.

Adım 2: Web Uygulamasını Başlatma

1.  Reflex Uygulamasını Çalıştırın:
    ```bash
    reflex run
    ```
Uygulama artık `http://localhost:3000` adresinde erişilebilir durumda!

---

## Gelecek Geliştirmeler
Faiss Entegrasyonu:** Milyonlarca görsellik veri setlerinde bile milisaniyeler içinde arama yapabilmek için `features.pkl` yerine Facebook'un Faiss kütüphanesini entegre etmek.
Metinle Görsel Arama:** Projeyi OpenAI'ın CLIP modelini kullanacak şekilde güncelleyerek "mavi bir bisiklet" gibi metinsel sorgularla görsel arama yeteneği eklemek.
UI İyileştirmeleri:** Sonuçlar için sayfalama (pagination) ve lazy loading eklemek.
