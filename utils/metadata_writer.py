"""
NeuralTrade - PDF Book Metadata Writer
======================================
PDF kitaplarına metadata eklemek için araç.
Metadata, RAG sisteminde akıllı filtreleme için kullanılır.

Kullanım:
    # Tüm kitapları listele
    python utils/metadata_writer.py --list
    
    # Metadata ekle/güncelle
    python utils/metadata_writer.py --file "book.pdf" --category "Forex" --author "John Doe" --year 2024
    
    # İnteraktif mod
    python utils/metadata_writer.py --interactive
    
    # Metadata görüntüle
    python utils/metadata_writer.py --show "book.pdf"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

# ============================================
# YAPILANDIRMA
# ============================================
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "/app/knowledge_base/books")
METADATA_FILE = "books_metadata.json"

# Desteklenen kategoriler
CATEGORIES = [
    "Forex",
    "Stocks", 
    "Crypto",
    "Derivatives",
    "Technical Analysis",
    "Fundamental Analysis",
    "Risk Management",
    "Trading Psychology",
    "Algorithmic Trading",
    "General"
]


def get_metadata_path() -> Path:
    """Metadata dosyasının yolunu döndürür."""
    return Path(KNOWLEDGE_BASE_PATH) / METADATA_FILE


def load_metadata() -> dict:
    """Mevcut metadata'yı yükler."""
    metadata_path = get_metadata_path()
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return {"books": {}, "last_updated": None}


def save_metadata(metadata: dict) -> None:
    """Metadata'yı kaydeder."""
    metadata["last_updated"] = datetime.now().isoformat()
    
    metadata_path = get_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"{Fore.GREEN}✅ Metadata kaydedildi: {metadata_path}{Style.RESET_ALL}")


def list_books() -> None:
    """Tüm PDF kitaplarını ve metadata'larını listeler."""
    books_path = Path(KNOWLEDGE_BASE_PATH)
    
    if not books_path.exists():
        print(f"{Fore.RED}❌ Klasör bulunamadı: {KNOWLEDGE_BASE_PATH}{Style.RESET_ALL}")
        return
    
    pdf_files = list(books_path.glob("*.pdf"))
    metadata = load_metadata()
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"📚 NEURALTRADE KİTAP KÜTÜPHANESİ")
    print(f"{'='*70}{Style.RESET_ALL}")
    print(f"📁 Konum: {KNOWLEDGE_BASE_PATH}")
    print(f"📄 Toplam: {len(pdf_files)} PDF\n")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        filename = pdf_file.name
        book_meta = metadata.get("books", {}).get(filename, {})
        
        # Metadata durumu
        if book_meta:
            status = f"{Fore.GREEN}✅ Metadata VAR{Style.RESET_ALL}"
            category = book_meta.get("category", "?")
            author = book_meta.get("author", "?")
            year = book_meta.get("year", "?")
        else:
            status = f"{Fore.YELLOW}⚠️ Metadata YOK{Style.RESET_ALL}"
            category = author = year = "-"
        
        # Dosya boyutu
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        
        print(f"{i}. {Fore.WHITE}{filename}{Style.RESET_ALL}")
        print(f"   {status}")
        print(f"   📊 Kategori: {category} | 👤 Yazar: {author} | 📅 Yıl: {year}")
        print(f"   💾 Boyut: {size_mb:.1f} MB")
        print()
    
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")


def show_book_metadata(filename: str) -> None:
    """Belirli bir kitabın metadata'sını gösterir."""
    metadata = load_metadata()
    book_meta = metadata.get("books", {}).get(filename, {})
    
    if not book_meta:
        print(f"{Fore.YELLOW}⚠️ '{filename}' için metadata bulunamadı.{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.CYAN}📖 {filename} - Metadata{Style.RESET_ALL}")
    print("-" * 50)
    for key, value in book_meta.items():
        print(f"  {key}: {value}")
    print()


def set_book_metadata(filename: str, category: str = None, author: str = None, 
                      year: int = None, tags: list = None, description: str = None) -> None:
    """Kitaba metadata ekler veya günceller."""
    
    # PDF'in varlığını kontrol et
    pdf_path = Path(KNOWLEDGE_BASE_PATH) / filename
    if not pdf_path.exists():
        print(f"{Fore.RED}❌ PDF bulunamadı: {filename}{Style.RESET_ALL}")
        return
    
    metadata = load_metadata()
    
    # Mevcut metadata'yı al veya yeni oluştur
    book_meta = metadata.get("books", {}).get(filename, {})
    
    # Güncelle
    if category:
        if category not in CATEGORIES:
            print(f"{Fore.YELLOW}⚠️ Geçersiz kategori. Desteklenen: {', '.join(CATEGORIES)}{Style.RESET_ALL}")
            return
        book_meta["category"] = category
        
    if author:
        book_meta["author"] = author
        
    if year:
        book_meta["year"] = year
        
    if tags:
        book_meta["tags"] = tags
        
    if description:
        book_meta["description"] = description
    
    # Otomatik alanlar
    book_meta["filename"] = filename
    book_meta["updated_at"] = datetime.now().isoformat()
    book_meta["file_size_mb"] = round(pdf_path.stat().st_size / (1024 * 1024), 2)
    
    # Kaydet
    if "books" not in metadata:
        metadata["books"] = {}
    metadata["books"][filename] = book_meta
    
    save_metadata(metadata)
    
    print(f"\n{Fore.GREEN}✅ Metadata güncellendi: {filename}{Style.RESET_ALL}")
    show_book_metadata(filename)


def interactive_mode() -> None:
    """İnteraktif metadata düzenleme modu."""
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"📝 NEURALTRADE METADATA DÜZENLEYICI (İnteraktif)")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    # Kitapları listele
    books_path = Path(KNOWLEDGE_BASE_PATH)
    pdf_files = list(books_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"{Fore.RED}❌ PDF bulunamadı!{Style.RESET_ALL}")
        return
    
    print("📚 Mevcut Kitaplar:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file.name}")
    
    print(f"\n{Fore.YELLOW}Kitap numarasını girin (0 = çıkış):{Style.RESET_ALL} ", end="")
    
    try:
        choice = int(input().strip())
        
        if choice == 0:
            print("👋 Çıkılıyor...")
            return
            
        if choice < 1 or choice > len(pdf_files):
            print(f"{Fore.RED}❌ Geçersiz seçim!{Style.RESET_ALL}")
            return
        
        selected_file = pdf_files[choice - 1].name
        print(f"\n📖 Seçilen: {selected_file}\n")
        
        # Kategori seç
        print("📊 Kategoriler:")
        for i, cat in enumerate(CATEGORIES, 1):
            print(f"  {i}. {cat}")
        
        print(f"\n{Fore.YELLOW}Kategori numarası:{Style.RESET_ALL} ", end="")
        cat_choice = int(input().strip())
        category = CATEGORIES[cat_choice - 1] if 1 <= cat_choice <= len(CATEGORIES) else None
        
        # Yazar
        print(f"{Fore.YELLOW}Yazar adı:{Style.RESET_ALL} ", end="")
        author = input().strip() or None
        
        # Yıl
        print(f"{Fore.YELLOW}Yayın yılı:{Style.RESET_ALL} ", end="")
        year_input = input().strip()
        year = int(year_input) if year_input else None
        
        # Açıklama
        print(f"{Fore.YELLOW}Kısa açıklama (opsiyonel):{Style.RESET_ALL} ", end="")
        description = input().strip() or None
        
        # Tags
        print(f"{Fore.YELLOW}Etiketler (virgülle ayır, opsiyonel):{Style.RESET_ALL} ", end="")
        tags_input = input().strip()
        tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
        
        # Kaydet
        set_book_metadata(
            filename=selected_file,
            category=category,
            author=author,
            year=year,
            description=description,
            tags=tags
        )
        
    except ValueError:
        print(f"{Fore.RED}❌ Geçersiz giriş!{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}İptal edildi.{Style.RESET_ALL}")


def auto_detect_metadata() -> None:
    """Tüm kitaplar için otomatik metadata oluşturur (dosya adından)."""
    
    books_path = Path(KNOWLEDGE_BASE_PATH)
    pdf_files = list(books_path.glob("*.pdf"))
    
    print(f"\n{Fore.CYAN}🤖 OTOMATİK METADATA TESPİTİ{Style.RESET_ALL}")
    print("-" * 50)
    
    for pdf_file in pdf_files:
        filename = pdf_file.name.lower()
        
        # Kategori tespiti
        category = "General"
        if any(word in filename for word in ["currency", "forex", "fx", "döviz"]):
            category = "Forex"
        elif any(word in filename for word in ["stock", "equity", "share", "hisse"]):
            category = "Stocks"
        elif any(word in filename for word in ["crypto", "bitcoin", "blockchain"]):
            category = "Crypto"
        elif any(word in filename for word in ["option", "derivative", "future"]):
            category = "Derivatives"
        elif any(word in filename for word in ["technical", "analysis", "chart", "teknik"]):
            category = "Technical Analysis"
        elif any(word in filename for word in ["risk", "management"]):
            category = "Risk Management"
        elif any(word in filename for word in ["psychology", "psikoloji"]):
            category = "Trading Psychology"
        elif any(word in filename for word in ["algo", "quant", "systematic"]):
            category = "Algorithmic Trading"
        
        # Yazar tespiti
        author = "Unknown"
        if "dummies" in filename:
            author = "For Dummies Series"
        elif "art-of" in filename or "art_of" in filename:
            author = "Expert Author"
        
        # Yıl (dosya adında varsa)
        import re
        year_match = re.search(r'(20\d{2})', filename)
        year = int(year_match.group(1)) if year_match else 2024
        
        set_book_metadata(
            filename=pdf_file.name,
            category=category,
            author=author,
            year=year
        )
    
    print(f"\n{Fore.GREEN}✅ {len(pdf_files)} kitap için otomatik metadata oluşturuldu.{Style.RESET_ALL}\n")


# ============================================
# ANA GİRİŞ NOKTASI
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="NeuralTrade PDF Book Metadata Writer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python utils/metadata_writer.py --list
  python utils/metadata_writer.py --interactive
  python utils/metadata_writer.py --auto-detect
  python utils/metadata_writer.py --file "book.pdf" --category "Forex" --author "John" --year 2024
  python utils/metadata_writer.py --show "book.pdf"
        """
    )
    
    parser.add_argument("--list", "-l", action="store_true", help="Tüm kitapları listele")
    parser.add_argument("--interactive", "-i", action="store_true", help="İnteraktif mod")
    parser.add_argument("--auto-detect", "-a", action="store_true", help="Otomatik metadata oluştur")
    parser.add_argument("--show", "-s", type=str, help="Kitabın metadata'sını göster")
    parser.add_argument("--file", "-f", type=str, help="PDF dosya adı")
    parser.add_argument("--category", "-c", type=str, choices=CATEGORIES, help="Kategori")
    parser.add_argument("--author", type=str, help="Yazar adı")
    parser.add_argument("--year", "-y", type=int, help="Yayın yılı")
    parser.add_argument("--description", "-d", type=str, help="Açıklama")
    parser.add_argument("--tags", "-t", type=str, help="Etiketler (virgülle ayır)")
    
    args = parser.parse_args()
    
    # Argüman yoksa yardım göster
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Komutları işle
    if args.list:
        list_books()
        
    elif args.interactive:
        interactive_mode()
        
    elif args.auto_detect:
        auto_detect_metadata()
        
    elif args.show:
        show_book_metadata(args.show)
        
    elif args.file:
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        set_book_metadata(
            filename=args.file,
            category=args.category,
            author=args.author,
            year=args.year,
            description=args.description,
            tags=tags
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
