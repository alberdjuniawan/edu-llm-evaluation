import json
from pathlib import Path

from edu_eval.generation.cases import load_cases, validation_report
from edu_eval.generation.config import GenerationConfig
from edu_eval.generation.schema import ControlledGenerationCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "controlled_generation_cases.jsonl"
GENERATION_CONFIG = PROJECT_ROOT / "configs" / "generation.yaml"

PHASE_SENTENCE = {
    "SD6": "Pemetaan fase resmi: Fase C untuk kelas V-VI SD.",
    "SMP7": "Pemetaan fase resmi: Fase D untuk kelas VII-IX SMP.",
    "SMP9": "Pemetaan fase resmi: Fase D untuk kelas VII-IX SMP.",
    "SMA10": "Pemetaan fase resmi: Fase E untuk kelas X SMA/SMK/MA.",
}

EVIDENCE_SOURCE = (
    "Project-authored grade-progression rationale (draft capsule) + "
    "CP Kurikulum Merdeka 2024 phase mapping "
    "(Fase C = V-VI SD, Fase D = VII-IX SMP, Fase E = X)"
)

REVIEW_NOTE = (
    "Rasional kemajuan tingkat kelas ini adalah ringkasan project-authored "
    "dan memerlukan telaah ahli sebelum evaluasi manusia."
)

# (subject, concept, capsule, sd6, smp7, smp9, sma10)
CASES: list[tuple[str, str, str, str, str, str, str]] = [
    # ---------------- IPA (12) ----------------
    (
        "IPA",
        "Fotosintesis",
        "Fotosintesis adalah proses tumbuhan hijau membuat makanan. Dengan bantuan cahaya matahari, daun menyerap air dari akar dan karbon dioksida dari udara, lalu menghasilkan gula (glukosa) dan oksigen. Klorofil, zat hijau daun, berperan menangkap energi cahaya.",
        "Peserta didik mengamati bahwa tumbuhan membutuhkan cahaya dan air untuk tumbuh.",
        "Peserta didik mengidentifikasi bahan (air, karbon dioksida, cahaya) dan hasil (gula, oksigen) fotosintesis.",
        "Peserta didik menjelaskan persamaan reaksi fotosintesis dan faktor yang memengaruhinya.",
        "Peserta didik menganalisis tahap terang dan tahap gelap serta peran spektrum cahaya.",
    ),
    (
        "IPA",
        "Rantai makanan dan ekosistem",
        "Ekosistem tersusun atas makhluk hidup dan lingkungannya yang saling berinteraksi. Rantai makanan menunjukkan perpindahan energi: produsen (tumbuhan) dimakan konsumen (hewan), dan pengurai (jamur, bakteri) menguraikan sisa makhluk hidup menjadi zat hara.",
        "Peserta didik menyebutkan contoh siapa makan siapa di sekitarnya.",
        "Peserta didik mengurutkan rantai makanan dan peran produsen, konsumen, dan pengurai.",
        "Peserta didik menjelaskan jaring-jaring makanan dan piramida energi.",
        "Peserta didik menganalisis dinamika populasi dan keseimbangan ekosistem bila satu komponen terganggu.",
    ),
    (
        "IPA",
        "Gaya dan gerak",
        "Gaya adalah dorongan atau tarikan yang dapat mengubah gerak benda, misalnya membuat benda diam menjadi bergerak atau mengubah arahnya. Gesekan adalah gaya yang menghambat gerak, contohnya rem sepeda yang menghentikan roda.",
        "Peserta didik memberi contoh dorongan dan tarikan dalam kehidupan sehari-hari.",
        "Peserta didik mengidentifikasi jenis-jenis gaya dan mengukur besar gaya secara sederhana.",
        "Peserta didik menjelaskan Hukum Newton tentang gerak beserta contohnya.",
        "Peserta didik menganalisis resultan gaya dan gerak lurus beraturan serta berubah beraturan.",
    ),
    (
        "IPA",
        "Energi dan perubahannya",
        "Energi adalah kemampuan untuk melakukan usaha dan terdapat dalam berbagai bentuk: panas, cahaya, gerak, listrik, dan kimia. Energi tidak dapat diciptakan atau dimusnahkan, hanya berubah bentuk, misalnya energi listrik menjadi cahaya pada lampu.",
        "Peserta didik menyebutkan contoh sumber energi di rumah dan sekolah.",
        "Peserta didik mengidentifikasi perubahan bentuk energi pada alat sehari-hari.",
        "Peserta didik menjelaskan hukum kekekalan energi dan efisiensi perubahan energi.",
        "Peserta didik menghitung usaha, energi, dan daya pada peristiwa fisika sederhana.",
    ),
    (
        "IPA",
        "Sistem pencernaan manusia",
        "Makanan dicerna mulai dari mulut (dikunyah dan dibasahi ludah), kerongkongan, lambung (diaduk dan dicerna asam serta enzim), usus halus (penyerapan sari makanan), hingga sisa makanan dibuang melalui usus besar dan anus.",
        "Peserta didik mengurutkan organ pencernaan dari mulut hingga anus.",
        "Peserta didik menjelaskan fungsi tiap organ dan peran enzim pencernaan.",
        "Peserta didik menjelaskan gangguan sistem pencernaan dan cara menjaga kesehatannya.",
        "Peserta didik menganalisis proses biokimia pencernaan dan penyerapan sari makanan di usus halus.",
    ),
    (
        "IPA",
        "Daur air",
        "Air di bumi terus berputar melalui daur air: air menguap karena panas matahari (evaporasi), uap air mendingin membentuk awan (kondensasi), lalu turun sebagai hujan (presipitasi) dan mengalir kembali ke sungai, danau, dan laut.",
        "Peserta didik mengurutkan tahapan daur air secara sederhana.",
        "Peserta didik menjelaskan peran panas matahari dalam penguapan dan pembentukan awan.",
        "Peserta didik menjelaskan air tanah, resapan, dan pentingnya konservasi air.",
        "Peserta didik menganalisis neraca air dan dampak perubahan iklim terhadap daur air.",
    ),
    (
        "IPA",
        "Tata surya",
        "Tata surya terdiri atas Matahari sebagai pusat dan delapan planet yang mengelilinginya: Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, dan Neptunus. Bumi berputar pada porosnya (rotasi) dan mengelilingi Matahari (revolusi).",
        "Peserta didik menyebutkan nama-nama planet dalam tata surya.",
        "Peserta didik menjelaskan rotasi dan revolusi Bumi serta akibatnya seperti siang-malam.",
        "Peserta didik menjelaskan gaya gravitasi, satelit alami dan buatan.",
        "Peserta didik menganalisis hukum Kepler dan skala jarak dalam tata surya.",
    ),
    (
        "IPA",
        "Listrik dan rangkaian sederhana",
        "Arus listrik mengalir dari kutub positif ke kutub negatif melalui rangkaian tertutup. Rangkaian seri menyusun komponen berderet sehingga jika satu lampu mati semuanya padam, sedangkan rangkaian paralel memberi tiap lampu jalur sendiri.",
        "Peserta didik menyebutkan bahaya listrik dan cara menghematnya.",
        "Peserta didik merangkai rangkaian listrik sederhana seri dan paralel.",
        "Peserta didik menjelaskan hukum Ohm tentang hubungan tegangan, arus, dan hambatan.",
        "Peserta didik menghitung daya dan energi listrik pada rangkaian.",
    ),
    (
        "IPA",
        "Klasifikasi makhluk hidup",
        "Makhluk hidup memiliki ciri: bernapas, bergerak, makan, tumbuh, berkembang biak, dan peka terhadap rangsang. Hewan bertulang belakang (vertebrata) meliputi ikan, amfibi, reptil, burung, dan mamalia; hewan tak bertulang belakang (invertebrata) misalnya serangga dan cacing.",
        "Peserta didik menyebutkan ciri-ciri makhluk hidup dari pengamatan.",
        "Peserta didik mengelompokkan hewan vertebrata dan invertebrata.",
        "Peserta didik menjelaskan tingkatan taksonomi dan kunci determinasi sederhana.",
        "Peserta didik menganalisis keanekaragaman hayati dan dasar kekerabatan makhluk hidup.",
    ),
    (
        "IPA",
        "Pencemaran lingkungan",
        "Pencemaran adalah masuknya zat berbahaya ke lingkungan sehingga kualitasnya menurun. Pencemaran udara berasal dari asap kendaraan dan pabrik, pencemaran air dari limbah dan sampah, serta pencemaran tanah dari plastik dan bahan kimia.",
        "Peserta didik memberi contoh perilaku membuang sampah pada tempatnya.",
        "Peserta didik mengidentifikasi jenis-jenis pencemaran di lingkungannya.",
        "Peserta didik menjelaskan dampak pencemaran bagi kesehatan dan ekosistem.",
        "Peserta didik menganalisis upaya mitigasi pencemaran dan peran kebijakan lingkungan.",
    ),
    (
        "IPA",
        "Suhu, kalor, dan pemuaian",
        "Suhu menyatakan derajat panas benda dan diukur dengan termometer dalam satuan derajat Celsius. Kalor adalah energi panas yang berpindah dari benda bersuhu tinggi ke bersuhu rendah, misalnya melalui konduksi pada sendok logam yang panas.",
        "Peserta didik membedakan benda panas dan dingin melalui indra dan contoh sehari-hari.",
        "Peserta didik mengukur suhu dengan termometer dan menjelaskan perpindahan kalor secara konduksi.",
        "Peserta didik menjelaskan kalor jenis, perubahan wujud zat, dan pemuaian.",
        "Peserta didik menerapkan asas Black dalam perhitungan pertukaran kalor.",
    ),
    (
        "IPA",
        "Perkembangbiakan makhluk hidup",
        "Makhluk hidup berkembang biak untuk melestarikan jenisnya. Tumbuhan berkembang biak secara generatif melalui biji (penyerbukan dan pembuahan) dan vegetatif seperti tunas, stek, dan cangkok. Hewan berkembang biak dengan bertelur, melahirkan, atau keduanya.",
        "Peserta didik memberi contoh hewan bertelur dan melahirkan.",
        "Peserta didik membedakan perkembangbiakan vegetatif dan generatif pada tumbuhan.",
        "Peserta didik menjelaskan penyerbukan, pembuahan, dan sistem reproduksi manusia secara dasar.",
        "Peserta didik menganalisis faktor yang memengaruhi keberhasilan reproduksi dan dasar bioteknologi reproduksi.",
    ),
    # ---------------- Bahasa Indonesia (12) ----------------
    (
        "Bahasa Indonesia",
        "Ide pokok paragraf",
        "Setiap paragraf memiliki ide pokok, yaitu gagasan utama yang dibahas, yang biasanya terletak pada kalimat utama. Kalimat lain dalam paragraf bersifat penjelas dan mendukung ide pokok tersebut.",
        "Peserta didik menemukan inti cerita atau bacaan pendek yang dibaca.",
        "Peserta didik menentukan kalimat utama dan ide pokok paragraf.",
        "Peserta didik membedakan ide pokok dan ide pendukung serta meringkas paragraf.",
        "Peserta didik menganalisis struktur gagasan dan merumuskan simpulan bacaan.",
    ),
    (
        "Bahasa Indonesia",
        "Kalimat efektif",
        "Kalimat efektif menyampaikan gagasan secara jelas, hemat kata, dan sesuai kaidah: memiliki subjek dan predikat yang jelas, tidak bermakna ganda, dan menggunakan kata baku serta tanda baca yang tepat.",
        "Peserta didik menyusun kalimat lengkap yang mudah dipahami.",
        "Peserta didik mengidentifikasi dan memperbaiki kalimat yang rancu.",
        "Peserta didik menerapkan syarat kalimat efektif dalam menulis paragraf.",
        "Peserta didik menganalisis keefektifan kalimat dalam teks argumentatif.",
    ),
    (
        "Bahasa Indonesia",
        "Teks deskripsi",
        "Teks deskripsi melukiskan objek seolah-olah pembaca melihat, mendengar, atau merasakannya sendiri, menggunakan kata sifat dan rincian indra. Strukturnya meliputi identifikasi, klasifikasi, dan deskripsi bagian.",
        "Peserta didik mendeskripsikan benda atau hewan kesukaan secara lisan dan tulis.",
        "Peserta didik menulis teks deskripsi dengan struktur dan kaidah yang tepat.",
        "Peserta didik mengembangkan deskripsi dengan majas dan pilihan kata yang tepat.",
        "Peserta didik menganalisis efek pilihan kata terhadap citraan dalam teks deskripsi.",
    ),
    (
        "Bahasa Indonesia",
        "Teks narasi",
        "Teks narasi menceritakan rangkaian peristiwa dengan unsur tokoh, latar, alur, dan amanat. Cerita imajinatif seperti dongeng dan fabel termasuk teks narasi yang bertujuan menghibur sekaligus menyampaikan pesan.",
        "Peserta didik menceritakan kembali dongeng atau pengalaman secara runtut.",
        "Peserta didik menulis cerita imajinatif dengan unsur intrinsik yang lengkap.",
        "Peserta didik mengembangkan konflik dan penyelesaian dalam cerita.",
        "Peserta didik menganalisis sudut pandang, gaya bahasa, dan amanat dalam narasi.",
    ),
    (
        "Bahasa Indonesia",
        "Puisi",
        "Puisi adalah karya sastra yang padat makna, terikat bait, larik, rima, dan irama. Puisi rakyat seperti pantun terikat aturan sampiran dan isi, sedangkan puisi modern lebih bebas namun tetap memperhatikan pilihan kata dan citraan.",
        "Peserta didik membaca dan menghafal pantun serta puisi sederhana.",
        "Peserta didik menulis pantun dan puisi pendek dengan rima.",
        "Peserta didik menganalisis unsur fisik dan batin puisi.",
        "Peserta didik menilai gaya bahasa, simbol, dan makna puisi modern.",
    ),
    (
        "Bahasa Indonesia",
        "Teks pidato",
        "Pidato adalah kegiatan berbicara di depan umum untuk menyampaikan gagasan. Teks pidato tersusun atas pembuka (salam dan sapaan), isi (pokok persoalan), dan penutup (simpulan dan ajakan), disampaikan dengan lafal, intonasi, dan sikap yang meyakinkan.",
        "Peserta didik berani berbicara menyampaikan pendapat di depan kelas.",
        "Peserta didik menyusun dan menyampaikan pidato singkat dengan struktur lengkap.",
        "Peserta didik berpidato persuasif dengan argumen dan bahasa yang santun.",
        "Peserta didik mengevaluasi retorika, argumen, dan etika berpidato.",
    ),
    (
        "Bahasa Indonesia",
        "Surat pribadi dan surat dinas",
        "Surat pribadi ditulis dengan bahasa santai kepada keluarga atau teman dan memuat kabar serta keperluan pribadi. Surat dinas bersifat resmi, memakai bahasa baku, kop, nomor, dan struktur baku untuk keperluan kedinasan.",
        "Peserta didik menulis surat pendek untuk teman atau keluarga.",
        "Peserta didik membedakan dan menulis surat pribadi serta surat dinas sederhana.",
        "Peserta didik menyusun surat lamaran dan surat resmi dengan kaidah lengkap.",
        "Peserta didik menganalisis keefektifan bahasa dan format surat resmi.",
    ),
    (
        "Bahasa Indonesia",
        "Teks eksplanasi",
        "Teks eksplanasi menjelaskan proses terjadinya suatu fenomena alam atau sosial secara sebab-akibat, misalnya terjadinya hujan atau banjir. Strukturnya meliputi pernyataan umum, deretan penjelas, dan interpretasi.",
        "Peserta didik menjelaskan urutan peristiwa sederhana dengan kata hubung waktu.",
        "Peserta didik menulis teks eksplanasi fenomena sehari-hari dengan struktur tepat.",
        "Peserta didik mengembangkan eksplanasi dengan data dan istilah ilmiah yang tepat.",
        "Peserta didik menilai kelengkapan kausalitas dan keakuratan penjelasan.",
    ),
    (
        "Bahasa Indonesia",
        "Kata baku",
        "Kata baku adalah kata yang sesuai dengan kaidah bahasa Indonesia dan tercantum dalam kamus, misalnya 'apotek' bukan 'apotik' dan 'mengubah' bukan 'merubah'. Penggunaan kata baku penting dalam ragam resmi dan karya tulis.",
        "Peserta didik membiasakan memakai kata yang benar dalam tulisan.",
        "Peserta didik mengidentifikasi dan membetulkan kata tidak baku dalam teks.",
        "Peserta didik menerapkan kata baku dan istilah dalam karya ilmiah sederhana.",
        "Peserta didik menganalisis pembentukan istilah dan penyerapan bahasa asing.",
    ),
    (
        "Bahasa Indonesia",
        "Imbuhan",
        "Imbuhan adalah bunyi yang ditambahkan pada kata dasar untuk membentuk kata baru: awalan (me-, ber-, di-), akhiran (-kan, -i, -an), sisipan (-el-, -em-), dan gabungan (ke-an, per-an). Imbuhan mengubah makna dan jenis kata.",
        "Peserta didik mengenali kata berimbuhan dalam bacaan.",
        "Peserta didik menentukan makna kata berimbuhan me-, ber-, di-, dan ter-.",
        "Peserta didik menganalisis fungsi imbuhan dan perubahannya dalam kalimat.",
        "Peserta didik mengevaluasi ketepatan pilihan imbuhan dalam ragam ilmiah.",
    ),
    (
        "Bahasa Indonesia",
        "Tanda baca dan huruf kapital",
        "Tanda baca seperti titik, koma, tanda tanya, dan tanda seru memperjelas makna kalimat. Huruf kapital dipakai di awal kalimat, nama orang, nama tempat, dan sapaan. Kesalahan tanda baca dapat mengubah makna.",
        "Peserta didik memakai titik dan huruf kapital pada awal kalimat.",
        "Peserta didik menerapkan koma, tanda tanya, dan tanda seru dengan tepat.",
        "Peserta didik menyunting teks dengan kaidah ejaan dan tanda baca yang lengkap.",
        "Peserta didik menganalisis ambiguitas makna akibat kesalahan tanda baca.",
    ),
    (
        "Bahasa Indonesia",
        "Iklan, slogan, dan poster",
        "Iklan membujuk khalayak memakai barang atau jasa melalui media, slogan adalah kalimat pendek yang mudah diingat, dan poster menyampaikan informasi secara visual dengan gambar dan teks singkat yang menarik.",
        "Peserta didik menyebutkan contoh iklan dan slogan yang dikenalnya.",
        "Peserta didik membuat slogan dan poster sederhana dengan bahasa persuasif.",
        "Peserta didik menganalisis strategi bahasa dan visual dalam iklan.",
        "Peserta didik menilai etika periklanan dan daya bujuk teks persuasif.",
    ),
    # ---------------- Pendidikan Pancasila (12) ----------------
    (
        "Pendidikan Pancasila",
        "Sila Ketuhanan Yang Maha Esa",
        "Sila pertama Pancasila menegaskan bangsa Indonesia bertuhan dan menjunjung kebebasan beragama. Pengamalannya antara lain beribadah sesuai agama masing-masing, menghormati pemeluk agama lain, dan tidak memaksakan kehendak beragama.",
        "Peserta didik menyebutkan contoh beribadah dan menghormati teman beda agama.",
        "Peserta didik menjelaskan makna sila pertama dan contoh pengamalannya.",
        "Peserta didik menganalisis hubungan sila pertama dengan kerukunan umat beragama.",
        "Peserta didik mengevaluasi kebijakan kebebasan beragama dalam bingkai Pancasila.",
    ),
    (
        "Pendidikan Pancasila",
        "Sila Kemanusiaan yang Adil dan Beradab",
        "Sila kedua menuntut kita memperlakukan sesama manusia secara adil dan beradab: menolong yang kesusahan, tidak semena-mena, menghargai pendapat, dan menjunjung nilai kemanusiaan tanpa membedakan suku atau agama.",
        "Peserta didik memberi contoh tolong-menolong kepada teman.",
        "Peserta didik menjelaskan makna adil dan beradab beserta contohnya.",
        "Peserta didik menganalisis pelanggaran kemanusiaan dan sikap yang seharusnya.",
        "Peserta didik mengevaluasi penegakan hak asasi manusia dalam kehidupan berbangsa.",
    ),
    (
        "Pendidikan Pancasila",
        "Sila Persatuan Indonesia",
        "Sila ketiga menegaskan bangsa Indonesia satu dan bersatu dalam keberagaman suku, agama, dan budaya. Pengamalannya antara lain bangga berbahasa Indonesia, mencintai produk dalam negeri, dan menolak perpecahan.",
        "Peserta didik menyebutkan contoh hidup rukun dengan teman berbeda suku.",
        "Peserta didik menjelaskan makna persatuan dan ancaman perpecahan.",
        "Peserta didik menganalisis peran persatuan dalam pembangunan nasional.",
        "Peserta didik mengevaluasi tantangan persatuan di era media sosial dan globalisasi.",
    ),
    (
        "Pendidikan Pancasila",
        "Sila Kerakyatan dan musyawarah",
        "Sila keempat menegaskan kedaulatan di tangan rakyat yang dilaksanakan melalui musyawarah untuk mufakat. Dalam musyawarah, setiap pendapat dihargai dan keputusan diambil demi kepentingan bersama, misalnya memilih ketua kelas.",
        "Peserta didik mengikuti pemilihan ketua kelas secara tertib.",
        "Peserta didik menjelaskan tata cara musyawarah dan menghargai pendapat.",
        "Peserta didik mempraktikkan musyawarah dalam pengambilan keputusan kelompok.",
        "Peserta didik menganalisis demokrasi Pancasila dan perwakilan dalam sistem ketatanegaraan.",
    ),
    (
        "Pendidikan Pancasila",
        "Sila Keadilan Sosial",
        "Sila kelima menuntut keadilan bagi seluruh rakyat Indonesia: setiap orang berhak mendapat perlakuan adil, kesempatan berusaha, dan perlindungan hukum. Contohnya pembagian tugas piket yang merata dan bantuan bagi yang kurang mampu.",
        "Peserta didik memberi contoh berbagi dan bersikap adil kepada teman.",
        "Peserta didik menjelaskan makna keadilan sosial dan contohnya.",
        "Peserta didik menganalisis ketimpangan sosial dan upaya mengatasinya.",
        "Peserta didik mengevaluasi kebijakan pemerataan pembangunan dan keadilan hukum.",
    ),
    (
        "Pendidikan Pancasila",
        "Gotong royong",
        "Gotong royong adalah bekerja sama tanpa pamrih untuk kepentingan bersama, misalnya kerja bakti membersihkan lingkungan, membantu tetangga hajatan, dan piket kelas. Gotong royong mencerminkan kepribadian dan budaya bangsa Indonesia.",
        "Peserta didik mengikuti piket kelas dan kerja bakti di sekolah.",
        "Peserta didik menjelaskan nilai dan manfaat gotong royong.",
        "Peserta didik menganalisis memudarnya gotong royong dan cara melestarikannya.",
        "Peserta didik mengevaluasi gotong royong sebagai modal sosial pembangunan.",
    ),
    (
        "Pendidikan Pancasila",
        "Hak dan kewajiban",
        "Setiap orang memiliki hak (sesuatu yang seharusnya diterima, misalnya mendapat pendidikan) dan kewajiban (sesuatu yang harus dilakukan, misalnya menaati aturan). Hak seseorang dibatasi oleh hak orang lain, sehingga keduanya harus seimbang.",
        "Peserta didik menyebutkan contoh hak dan kewajiban di rumah dan sekolah.",
        "Peserta didik menjelaskan dan melaksanakan hak serta kewajiban sebagai warga sekolah.",
        "Peserta didik menganalisis pelanggaran hak dan pengingkaran kewajiban di masyarakat.",
        "Peserta didik mengevaluasi jaminan hak warga negara dalam konstitusi.",
    ),
    (
        "Pendidikan Pancasila",
        "Norma dan aturan",
        "Norma adalah pedoman perilaku yang berlaku di masyarakat: norma agama, kesusilaan, kesopanan, dan hukum. Aturan tertulis seperti tata tertib sekolah wajib ditaati, dan pelanggarannya mendapat sanksi agar tercipta ketertiban.",
        "Peserta didik menyebutkan contoh aturan di rumah dan sekolah.",
        "Peserta didik menjelaskan macam norma dan sanksi beserta contohnya.",
        "Peserta didik menganalisis pentingnya kepatuhan hukum bagi ketertiban.",
        "Peserta didik mengevaluasi kesadaran hukum warga dan penegakannya.",
    ),
    (
        "Pendidikan Pancasila",
        "Bhinneka Tunggal Ika",
        "Semboyan Bhinneka Tunggal Ika berarti berbeda-beda tetapi tetap satu jua, menggambarkan persatuan Indonesia di tengah keragaman suku, bahasa, agama, dan budaya. Semboyan ini tertulis pada pita yang dicengkeram Garuda Pancasila.",
        "Peserta didik menyebutkan contoh keragaman suku dan budaya di Indonesia.",
        "Peserta didik menjelaskan makna semboyan dan contoh sikap menghargai keragaman.",
        "Peserta didik menganalisis konflik keberagaman dan cara penyelesaiannya.",
        "Peserta didik mengevaluasi politik kebhinnekaan dalam kehidupan berbangsa.",
    ),
    (
        "Pendidikan Pancasila",
        "Undang-Undang Dasar 1945",
        "UUD 1945 adalah hukum dasar tertulis negara Indonesia yang menjadi sumber segala peraturan. UUD 1945 memuat pembukaan, batang tubuh, dan penjelasan, serta menetapkan bentuk negara kesatuan, kedaulatan rakyat, dan jaminan hak warga negara.",
        "Peserta didik mengetahui bahwa Indonesia memiliki aturan dasar negara.",
        "Peserta didik menjelaskan kedudukan dan fungsi UUD 1945 secara sederhana.",
        "Peserta didik menjelaskan sistematika dan isi pokok UUD 1945.",
        "Peserta didik menganalisis hubungan konstitusi dengan peraturan perundang-undangan.",
    ),
    (
        "Pendidikan Pancasila",
        "Demokrasi Pancasila",
        "Demokrasi Pancasila adalah pemerintahan dari, oleh, dan untuk rakyat yang berlandaskan nilai Pancasila: mengutamakan musyawarah, menghargai hak minoritas, dan menjunjung hukum. Pemilu adalah wujud nyata kedaulatan rakyat.",
        "Peserta didik menyampaikan pendapat dan menghargai hasil voting kelas.",
        "Peserta didik menjelaskan ciri demokrasi dan contoh pelaksanaannya di sekolah.",
        "Peserta didik menganalisis pelaksanaan pemilu dan peran warga negara.",
        "Peserta didik mengevaluasi kualitas demokrasi dan partisipasi politik warga.",
    ),
    (
        "Pendidikan Pancasila",
        "Cinta tanah air",
        "Cinta tanah air diwujudkan dengan bangga menjadi bangsa Indonesia: memakai bahasa Indonesia, melestarikan budaya, menjaga lingkungan, berprestasi, dan rela berkorban membela negara. Upacara bendera menumbuhkan nasionalisme.",
        "Peserta didik mengikuti upacara bendera dengan tertib dan khidmat.",
        "Peserta didik menjelaskan wujud cinta tanah air di lingkungan sekolah.",
        "Peserta didik menganalisis ancaman nasionalisme dan cara menangkalnya.",
        "Peserta didik mengevaluasi bela negara dalam konteks warga modern.",
    ),
    # ---------------- Informatika (12) ----------------
    (
        "Informatika",
        "Data dan informasi",
        "Data adalah fakta mentah yang belum diolah, misalnya angka nilai ulangan. Setelah diolah dan disusun bermakna, data menjadi informasi, misalnya rata-rata kelas yang dipakai guru untuk menilai pemahaman siswa.",
        "Peserta didik memberi contoh data sederhana seperti nama dan angka.",
        "Peserta didik membedakan data dan informasi dari contoh sehari-hari.",
        "Peserta didik menjelaskan pengolahan data menjadi informasi yang berguna.",
        "Peserta didik menganalisis kualitas informasi dan pengambilan keputusan berbasis data.",
    ),
    (
        "Informatika",
        "Algoritma",
        "Algoritma adalah urutan langkah logis untuk menyelesaikan masalah, misalnya resep masakan atau tata cara menyalakan komputer. Algoritma yang baik bersifat jelas, tepat, dan berakhir (tidak berputar tanpa henti).",
        "Peserta didik menyusun urutan kegiatan harian secara runtut.",
        "Peserta didik menulis langkah-langkah penyelesaian masalah sederhana.",
        "Peserta didik menyusun algoritma dengan percabangan dan pengulangan.",
        "Peserta didik menganalisis efisiensi langkah dan merancang algoritma terstruktur.",
    ),
    (
        "Informatika",
        "Perangkat keras komputer",
        "Perangkat keras (hardware) adalah bagian fisik komputer: perangkat masukan (keyboard, mouse), pemroses (CPU), penyimpan (hard disk, flash disk), dan keluaran (monitor, printer). Setiap bagian memiliki fungsi yang saling melengkapi.",
        "Peserta didik menyebutkan nama bagian-bagian komputer.",
        "Peserta didik mengidentifikasi fungsi tiap perangkat keras.",
        "Peserta didik menjelaskan cara kerja dan perawatan perangkat keras.",
        "Peserta didik menganalisis spesifikasi perangkat untuk kebutuhan tertentu.",
    ),
    (
        "Informatika",
        "Perangkat lunak dan aplikasi",
        "Perangkat lunak (software) adalah program yang menjalankan komputer, misalnya sistem operasi, peramban, dan aplikasi pengolah kata. Tanpa perangkat lunak, perangkat keras hanyalah besi yang tidak berguna.",
        "Peserta didik menyebutkan contoh aplikasi yang dikenalnya.",
        "Peserta didik membedakan sistem operasi dan aplikasi serta menggunakannya.",
        "Peserta didik menjelaskan jenis lisensi dan pemasangan perangkat lunak.",
        "Peserta didik mengevaluasi pemilihan perangkat lunak yang legal dan sesuai kebutuhan.",
    ),
    (
        "Informatika",
        "Dekomposisi dalam berpikir komputasional",
        "Dekomposisi adalah memecah masalah besar menjadi bagian-bagian kecil yang mudah diselesaikan, misalnya membersihkan rumah dipecah menjadi menyapu, mengepel, dan merapikan. Ini adalah fondasi berpikir komputasional.",
        "Peserta didik memecah tugas besar menjadi langkah kecil.",
        "Peserta didik menerapkan dekomposisi pada masalah sehari-hari.",
        "Peserta didik menggabungkan dekomposisi dengan pengenalan pola dan abstraksi.",
        "Peserta didik merancang solusi komputasional dengan dekomposisi sistematis.",
    ),
    (
        "Informatika",
        "Jaringan komputer dan internet",
        "Jaringan menghubungkan beberapa komputer agar dapat berbagi data dan perangkat, misalnya jaringan sekolah. Internet adalah jaringan raksasa antarkomputer di seluruh dunia yang memungkinkan surat elektronik, belajar daring, dan pencarian informasi.",
        "Peserta didik mengetahui bahwa internet menghubungkan banyak orang.",
        "Peserta didik menjelaskan manfaat dan cara aman memakai internet.",
        "Peserta didik menjelaskan topologi dan perangkat jaringan secara dasar.",
        "Peserta didik menganalisis arsitektur jaringan dan layanan internet.",
    ),
    (
        "Informatika",
        "Keamanan digital",
        "Keamanan digital melindungi data dan akun dari pencurian: memakai kata sandi kuat dan berbeda, tidak membagikan kode OTP, waspada tautan asing, dan memperbarui perangkat lunak. Data pribadi seperti NISN tidak boleh disebar sembarangan.",
        "Peserta didik menjaga kerahasiaan kata sandi dan data pribadi.",
        "Peserta didik menjelaskan ancaman umum seperti phising dan cara menghindarinya.",
        "Peserta didik menerapkan autentikasi ganda dan kebiasaan keamanan berlapis.",
        "Peserta didik menganalisis serangan siber dan strategi mitigasinya.",
    ),
    (
        "Informatika",
        "Etika digital",
        "Etika digital mengatur perilaku baik di dunia maya: memakai bahasa santun, menghargai karya orang (tidak menyalin tanpa izin), tidak menyebar hoaks dan ujaran kebencian, serta menghormati privasi orang lain.",
        "Peserta didik berbahasa santun saat berkirim pesan.",
        "Peserta didik menjelaskan aturan sopan santun dan hak cipta di internet.",
        "Peserta didik menganalisis kasus pelanggaran etika digital dan akibatnya.",
        "Peserta didik mengevaluasi dilema etika teknologi seperti privasi dan AI.",
    ),
    (
        "Informatika",
        "Representasi data",
        "Komputer menyimpan semua data sebagai bilangan biner (0 dan 1). Huruf, angka, gambar, dan suara dikodekan menjadi pola bit, misalnya huruf A dikodekan menurut tabel standar sehingga komputer dapat menampilkannya kembali.",
        "Peserta didik mengetahui komputer bekerja dengan angka nol dan satu.",
        "Peserta didik menjelaskan konversi bilangan desimal dan biner sederhana.",
        "Peserta didik menjelaskan pengodean teks, gambar, dan satuan ukuran data.",
        "Peserta didik menganalisis kapasitas penyimpanan dan kompresi data.",
    ),
    (
        "Informatika",
        "Aplikasi pengolah kata dan angka",
        "Aplikasi pengolah kata dipakai menulis dokumen dengan huruf, paragraf, tabel, dan gambar, sedangkan pengolah angka (lembar kerja) menghitung dengan rumus pada sel baris dan kolom, misalnya menjumlah nilai dengan fungsi SUM.",
        "Peserta didik mengetik dan menyimpan dokumen sederhana.",
        "Peserta didik membuat dokumen rapi dan tabel perhitungan sederhana.",
        "Peserta didik memakai rumus, diagram, dan format pada lembar kerja.",
        "Peserta didik mengotomatisasi laporan dengan rumus lanjutan dan validasi data.",
    ),
    (
        "Informatika",
        "Literasi informasi dan hoaks",
        "Tidak semua informasi di internet benar. Literasi informasi berarti memeriksa sumber, tanggal, dan bukti sebelum percaya dan membagikan. Hoaks adalah kabar bohong yang sengaja disebar dan dapat merugikan banyak orang.",
        "Peserta didik bertanya kepada guru sebelum mempercayai kabar aneh.",
        "Peserta didik memeriksa kebenaran informasi dengan bertanya dan membandingkan sumber.",
        "Peserta didik memverifikasi berita dengan memeriksa sumber primer dan fakta.",
        "Peserta didik menganalisis teknik misinformasi dan strategi memeranginya.",
    ),
    (
        "Informatika",
        "Dampak sosial informatika",
        "Teknologi informasi mengubah cara belajar, bekerja, dan bergaul: memudahkan akses ilmu tetapi juga menimbulkan kecanduan gawai, berkurangnya interaksi langsung, dan kesenjangan bagi yang tak memiliki akses (kesenjangan digital).",
        "Peserta didik menyebutkan manfaat dan bahaya gawai secara seimbang.",
        "Peserta didik menjelaskan dampak positif dan negatif teknologi di sekitarnya.",
        "Peserta didik menganalisis kecanduan gawai dan kesenjangan digital.",
        "Peserta didik mengevaluasi kebijakan teknologi dan tanggung jawab sosial pengguna.",
    ),
    # ---------------- IPS (12) ----------------
    (
        "IPS",
        "Kenampakan alam dan peta",
        "Kenampakan alam meliputi dataran rendah, dataran tinggi, gunung, lembah, sungai, danau, dan laut. Peta menggambarkan permukaan bumi pada bidang datar dengan judul, legenda, skala, dan mata angin sebagai petunjuk membacanya.",
        "Peserta didik menyebutkan contoh gunung, sungai, dan laut di Indonesia.",
        "Peserta didik membaca peta sederhana dengan legenda dan mata angin.",
        "Peserta didik menjelaskan skala peta dan menghitung jarak sebenarnya.",
        "Peserta didik menganalisis hubungan bentang alam dengan kehidupan penduduk.",
    ),
    (
        "IPS",
        "Sumber daya alam",
        "Sumber daya alam adalah kekayaan dari alam yang dimanfaatkan manusia: dapat diperbarui (tumbuhan, hewan, air, udara) dan tidak dapat diperbarui (minyak, batu bara, emas). Pemakaiannya harus hemat dan lestari agar tidak habis.",
        "Peserta didik menyebutkan contoh hasil alam di daerahnya.",
        "Peserta didik membedakan sumber daya dapat dan tidak dapat diperbarui.",
        "Peserta didik menjelaskan persebaran sumber daya alam Indonesia dan pemanfaatannya.",
        "Peserta didik menganalisis pembangunan berkelanjutan dan krisis sumber daya.",
    ),
    (
        "IPS",
        "Kegiatan ekonomi",
        "Kegiatan ekonomi meliputi produksi (membuat barang/jasa), distribusi (menyalurkan), dan konsumsi (memakai). Petani memproduksi padi, pedagang mendistribusikannya ke pasar, dan keluarga mengonsumsinya sebagai nasi.",
        "Peserta didik menyebutkan pekerjaan orang tua dan barang yang dihasilkan.",
        "Peserta didik menjelaskan rantai produksi, distribusi, dan konsumsi.",
        "Peserta didik menganalisis pelaku ekonomi dan perannya dalam pasar.",
        "Peserta didik mengevaluasi sistem ekonomi dan interdependensi antarpelaku.",
    ),
    (
        "IPS",
        "Uang dan pasar",
        "Uang berfungsi sebagai alat tukar, satuan hitung, dan penyimpan nilai. Pasar adalah tempat bertemunya penjual dan pembeli; harga terbentuk dari tawar-menawar berdasarkan permintaan dan penawaran.",
        "Peserta didik mengenal nilai pecahan uang dan berbelanja sederhana.",
        "Peserta didik menjelaskan fungsi uang dan jenis-jenis pasar.",
        "Peserta didik menjelaskan mekanisme permintaan, penawaran, dan harga.",
        "Peserta didik menganalisis inflasi, perbankan, dan pasar modern.",
    ),
    (
        "IPS",
        "Interaksi sosial",
        "Interaksi sosial adalah hubungan timbal balik antarmanusia yang didorong kebutuhan bekerja sama. Bentuknya asosiatif (kerja sama, akomodasi) dan disosiatif (persaingan, konflik). Aturan dan norma menjaga interaksi tetap tertib.",
        "Peserta didik bermain dan bekerja sama dengan teman.",
        "Peserta didik menjelaskan syarat dan bentuk interaksi sosial.",
        "Peserta didik menganalisis faktor pendorong dan penghambat interaksi.",
        "Peserta didik mengevaluasi konflik sosial dan strategi resolusinya.",
    ),
    (
        "IPS",
        "Keragaman budaya Indonesia",
        "Indonesia kaya budaya: ratusan suku dengan bahasa daerah, rumah adat, pakaian adat, tarian, lagu daerah, dan makanan khas masing-masing. Keragaman ini adalah kekayaan yang harus dihormati dan dilestarikan, bukan dipertentangkan.",
        "Peserta didik menyebutkan contoh lagu daerah dan makanan khas.",
        "Peserta didik menjelaskan wujud keragaman budaya dan cara menghargainya.",
        "Peserta didik menganalisis akulturasi budaya dan pelestarian warisan.",
        "Peserta didik mengevaluasi integrasi nasional di tengah keragaman budaya.",
    ),
    (
        "IPS",
        "Proklamasi kemerdekaan",
        "Kemerdekaan Indonesia diproklamasikan pada 17 Agustus 1945 oleh Soekarno-Hatta atas perjuangan panjang melawan penjajah. Sehari sesudahnya, PPKI mengesahkan UUD 1945 dan memilih presiden, meletakkan dasar negara merdeka.",
        "Peserta didik mengetahui hari kemerdekaan dan para pahlawan.",
        "Peserta didik menjelaskan rangkaian peristiwa proklamasi secara runtut.",
        "Peserta didik menganalisis makna proklamasi dan perjuangan mempertahankannya.",
        "Peserta didik mengevaluasi historiografi dan nilai perjuangan bagi generasi kini.",
    ),
    (
        "IPS",
        "ASEAN dan kerja sama regional",
        "ASEAN adalah perhimpunan bangsa-bangsa Asia Tenggara yang didirikan 1967 di Bangkok, beranggotakan antara lain Indonesia, Malaysia, Singapura, Thailand, dan Filipina. Tujuannya memajukan kerja sama ekonomi, sosial, dan perdamaian kawasan.",
        "Peserta didik menyebutkan negara tetangga Indonesia.",
        "Peserta didik menjelaskan latar berdirinya dan anggota ASEAN.",
        "Peserta didik menjelaskan bentuk kerja sama dan peran Indonesia di ASEAN.",
        "Peserta didik menganalisis tantangan integrasi kawasan dan posisi Indonesia.",
    ),
    (
        "IPS",
        "Transportasi dan komunikasi",
        "Transportasi memindahkan orang dan barang melalui darat, laut, dan udara; komunikasi menyampaikan pesan melalui surat, telepon, hingga internet. Kemajuan keduanya memperlancar perdagangan dan mempersempit jarak antarwilayah.",
        "Peserta didik menyebutkan alat transportasi yang dikenalnya.",
        "Peserta didik menjelaskan jenis transportasi dan komunikasi serta fungsinya.",
        "Peserta didik menjelaskan perkembangan teknologi transportasi dan komunikasi.",
        "Peserta didik menganalisis infrastruktur dan pemerataan konektivitas wilayah.",
    ),
    (
        "IPS",
        "Kependudukan dan mobilitas",
        "Penduduk bertambah melalui kelahiran dan berkurang karena kematian, sedangkan perpindahan (migrasi) mengubah sebaran, misalnya urbanisasi dari desa ke kota. Kepadatan dan komposisi penduduk memengaruhi kebutuhan sekolah, lapangan kerja, dan layanan.",
        "Peserta didik mengetahui desanya padat atau jarang penduduk.",
        "Peserta didik menjelaskan kelahiran, kematian, dan perpindahan penduduk.",
        "Peserta didik menjelaskan piramida penduduk dan masalah kependudukan.",
        "Peserta didik menganalisis bonus demografi dan kebijakan kependudukan.",
    ),
    (
        "IPS",
        "Potensi daerah",
        "Setiap daerah memiliki potensi unggulan: pertanian di dataran subur, perikanan di pesisir, pariwisata di kawasan indah, dan industri di kota besar. Pengembangan potensi daerah meningkatkan pendapatan dan kesejahteraan warganya.",
        "Peserta didik menyebutkan hasil utama daerahnya.",
        "Peserta didik mengidentifikasi potensi daerahnya dan cara memanfaatkannya.",
        "Peserta didik menjelaskan keunggulan komparatif dan pengembangan wilayah.",
        "Peserta didik mengevaluasi otonomi daerah dan pemerataan pembangunan.",
    ),
    (
        "IPS",
        "Globalisasi",
        "Globalisasi adalah menyatunya dunia melalui perdagangan, teknologi, dan budaya sehingga batas negara makin kabur: produk asing mudah dibeli, kabar dunia cepat tersebar, tetapi budaya lokal terancam dan persaingan makin ketat.",
        "Peserta didik menyebutkan contoh barang dan tontonan dari luar negeri.",
        "Peserta didik menjelaskan pengertian dan contoh globalisasi di sekitarnya.",
        "Peserta didik menjelaskan dampak positif dan negatif globalisasi.",
        "Peserta didik mengevaluasi strategi Indonesia menghadapi persaingan global.",
    ),
]


def build_case(
    index: int,
    subject: str,
    concept: str,
    capsule: str,
    notes: dict[str, str],
) -> dict:
    case_id = f"CG-{index:03d}"
    grades = ["SD6", "SMP7", "SMP9", "SMA10"]
    phases = ["C", "D", "D", "E"]
    grade_numbers = [6, 7, 9, 10]

    target_evidence = {}

    for target, phase, grade_number in zip(grades, phases, grade_numbers, strict=True):
        target_evidence[target] = {
            "curriculum_phase": phase,
            "grade": grade_number,
            "evidence_source": EVIDENCE_SOURCE,
            "evidence_locator": None,
            "evidence_text": (
                f"{notes[target]} {PHASE_SENTENCE[target]} {REVIEW_NOTE}"
            ),
            "evidence_scope": "phase_only",
        }

    return {
        "case_id": case_id,
        "subject": subject,
        "concept": concept,
        "task_type": "materi",
        "reference": {
            "source_id": f"REF-{case_id}",
            "source_title": (
                f"Kapsul referensi project-authored - {concept} ({subject}) [draf]"
            ),
            "source_url": None,
            "source_type": "project_capsule",
            "source_version": "capsule-v1-draft",
            "locator": None,
            "text": capsule,
        },
        "target_evidence": target_evidence,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payloads = [
        build_case(
            index,
            subject,
            concept,
            capsule,
            {"SD6": sd6, "SMP7": smp7, "SMP9": smp9, "SMA10": sma10},
        )
        for index, (
            subject,
            concept,
            capsule,
            sd6,
            smp7,
            smp9,
            sma10,
        ) in enumerate(CASES, start=1)
    ]

    # Fail loudly before writing anything.
    validated = [
        ControlledGenerationCase.model_validate(payload) for payload in payloads
    ]

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for payload in payloads:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"{len(payloads)} cases -> {OUTPUT_PATH}")

    cases = load_cases(OUTPUT_PATH)
    config = GenerationConfig.from_yaml(GENERATION_CONFIG)
    report = validation_report(cases, config)

    n_subjects = len(report["subjects"])
    print(f"subjects={report['subjects']}")
    print(
        f"pilot={config.pilot_cases_per_subject * n_subjects} "
        f"full={config.full_cases_per_subject * n_subjects}"
    )
    print(
        f"schema={report['schema']} provenance={report['provenance']} "
        f"phase={report['phase_mapping']} dup={report['duplicate_case_ids']} "
        f"pilot={report['pilot_ready']} full={report['full_ready']}"
    )

    for note in report["notes"]:
        print(f"note: {note}")

    assert len(validated) == 60, "Spec requires exactly 60 canonical cases."


if __name__ == "__main__":
    main()
