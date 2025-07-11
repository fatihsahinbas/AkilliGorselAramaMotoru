import reflex as rx
import os
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from .feature_extractor import FeatureExtractor

# İndekslenmiş verileri ve özellik çıkarıcıyı yükle
try:
    with open('features.pkl', 'rb') as f:
        INDEX_DATA = pickle.load(f)
    EXTRACTOR = FeatureExtractor()
    IS_READY = True
except FileNotFoundError:
    INDEX_DATA = None
    EXTRACTOR = None
    IS_READY = False

class State(rx.State):
    is_ready: bool = IS_READY
    is_processing: bool = False
    uploaded_image_path: str = ""
    results: list[str] = []

    async def handle_upload(self, files):
        if not files:
            return
        
        self.is_processing = True
        self.results = []
        
        upload_data = files[0]
        
        temp_filename = "uploaded_image.jpg"
        
        output_path = os.path.join("assets", temp_filename)
        
        with open(output_path, "wb") as f:
            f.write(upload_data)
        
        self.uploaded_image_path = temp_filename
        
        # Arama yap
        self.perform_search(output_path)
        self.is_processing = False

    def perform_search(self, query_path: str):
        with Image.open(query_path) as img:
            query_vector = EXTRACTOR.extract(img)
        
        # Kosinüs benzerliklerini hesapla
        similarities = cosine_similarity([query_vector], INDEX_DATA['features'])[0]
        
        # En benzer N sonucu bul (kendisi hariç)
        top_n = 10 
        # benzerlik skorlarına göre sırala ve en yüksek olanları al
        indices = similarities.argsort()[::-1][1:top_n+1]
        
        self.results = [INDEX_DATA['paths'][i] for i in indices]

def index():
    return rx.container(
        rx.vstack(
            rx.heading("Görsel Arama Motoru", size="8"),
            rx.text("Benzerini bulmak istediğiniz bir görsel yükleyin."),
            
            rx.cond(
                State.is_ready,
                rx.vstack(
                    rx.upload(
                        rx.button("Görsel Seç", is_disabled=State.is_processing),
                        rx.text("veya sürükleyip bırakın."),
                        id="upload_area",
                        # DOĞRU KULLANIM:
                        on_drop=State.handle_upload,
                        border="2px dashed #ccc",
                        padding="2em",
                        width="100%"
                    ),
                    rx.cond(
                        State.is_processing,
                        rx.center(rx.spinner(), margin_top="2em")
                    ),
                    rx.cond(
                        State.uploaded_image_path,
                        rx.vstack(
                            rx.heading("Sorgu Görseli", size="6", margin_top="2em"),
                            rx.image(src=State.uploaded_image_path, width="200px", height="auto", border="2px solid #ddd", padding="0.5em"),
                            align="center"
                        )
                    ),
                    rx.cond(
                        State.results,
                        rx.vstack(
                            rx.heading("Benzer Sonuçlar", size="6", margin_top="2em"),
                            rx.grid(
                                rx.foreach(
                                    State.results,
                                    lambda path: rx.image(src=path, width="150px")
                                ),
                                columns="5",
                                spacing="4"
                            ),
                            align="center"
                        )
                    )
                ),
                # Eğer index dosyası yoksa gösterilecek uyarı
                rx.callout(
                    "Sistem hazır değil. Lütfen 'python indexer.py' komutunu çalıştırarak görselleri indeksleyin.",
                    icon="alert_triangle",
                    color_scheme="red",
                    role="alert",
                    width="100%",
                    margin_top="2rem"
                )
            ),
            spacing="5",
            padding_y="3em",
            align="center"
        )
    )

app = rx.App()
app.add_page(index, title="Görsel Arama Motoru")