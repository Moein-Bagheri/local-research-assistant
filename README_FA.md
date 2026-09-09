# دستیار پژوهشی محلی برای چهار مقاله RAG

[English overview](README.md)

این پروژه یک دستیار پژوهشی کامل و محلی برای چهار مقاله‌ی `REALM`، `DPR`، `RAG` و `FiD` است. برنامه PDFها را استخراج و قطعه‌بندی می‌کند، embedding محلی می‌سازد، بردارها را در FAISS قرار می‌دهد، با RAG پاسخ دارای شماره صفحه تولید می‌کند، درخواست‌های خاص را با Function Calling به شش تابع هدایت می‌کند و در نبود مدرک کافی به پاسخ عمومی مدل محلی می‌رود.

هیچ API پولی استفاده نشده است. مدل تولید `qwen2.5:1.5b` و مدل embedding برابر `all-minilm` است و هر دو با Ollama روی سیستم اجرا می‌شوند.

## تصمیم علمی درباره مقالات

چهار فایل اولیه شامل REALM، RAG، FiD و Self-RAG بودند. بررسی دقیق نشان داد Self-RAG روی TriviaQA معیار `answer-containment accuracy` را به‌کار می‌برد، درحالی‌که FiD معیار سخت‌گیرانه‌ی `Exact Match` دارد؛ همچنین هیچ دیتاست آزمون واحدی میان هر چهار فایل اولیه نبود. به همین دلیل مقاله‌ی رسمی DPR از ACL Anthology افزوده شد و هسته‌ی عددی پروژه به این چهار مقاله تغییر کرد:

| مقاله | نسخه‌ی گزارش‌شده | Natural Questions test EM |
|---|---|---:|
| REALM | REALM-CC-News | 40.4 |
| DPR | Single DPR | 41.5 |
| RAG | RAG-Sequence | 44.5 |
| FiD | FiD-large | 51.4 |

هر چهار مقدار روی Natural Questions در تنظیم open-domain و معیار Exact Match گزارش شده‌اند. بااین‌حال مدل، تعداد passage و بودجه‌ی retrieval یکسان نیست؛ پس این جدول یک مقایسه‌ی benchmark است، نه ablation کنترل‌شده. Self-RAG در پوشه باقی مانده، ولی برنامه عمداً آن را در مقایسه‌ی عددی اصلی وارد نمی‌کند.

## نگاشت مستقیم به مباحث درس

- پیش‌پردازش: پاک‌سازی PDF، تحلیل طول chunk با Histogram، Z-score و IQR.
- Chunking و Embedding: پنجره‌ی ۸۰ کلمه‌ای با ۱۵ کلمه هم‌پوشانی و embedding محلی ۳۸۴بعدی.
- معیار فاصله: L2 normalization و inner product در FAISS که برابر cosine similarity است.
- کاهش ابعاد: PCA دوبعدی فقط برای تحلیل بصری embeddingها.
- خوشه‌بندی: K-Means، elbow و Silhouette Score فقط به‌عنوان تحلیل جانبی.
- LLM و RAG: Qwen2.5-1.5B، retrieval top-3، context و پاسخ دارای منبع.
- Function Calling: مسیریابی خودکار مدل به سه تابع اجباری و سه تابع امتیازی.
- ارزیابی: Accuracy، Precision و F1 برای retrieval/OOD/routing و Token F1 برای پاسخ کوتاه.

ابزارهایی مانند Regex، Matplotlib و Streamlit تکنیک داده‌کاوی جدیدی به پروژه اضافه نمی‌کنند؛ Regex فقط برای تابع اختیاری شمارش سال منابع، Matplotlib برای رسم نمودار و Streamlit صرفاً رابط نمایشی مجاز صورت‌مسئله است.

## معماری و انتخاب‌ها

1. `PyMuPDF` متن PDF را صفحه‌به‌صفحه استخراج می‌کند و headingهای Abstract، Introduction، Method، Experiments، Results، Conclusion و References را نگه می‌دارد.
2. هر بخش به chunkهای ۸۰ کلمه‌ای با overlap برابر ۱۵ تقسیم می‌شود. مقدار ۸۰ پس از آزمون عملی انتخاب شد، چون context مدل `all-minilm` برای chunkهای ۱۰۰ تا ۱۸۰ کلمه‌ایِ دارای فرمول خطای طول می‌داد. overlap از قطع‌شدن مفهوم روی مرز جلوگیری می‌کند.
3. ۴۸۰ chunk خام PDF ساخته شد. چهار رکورد نتیجه و چهار رکورد روش نیز به‌صورت human-verified و دارای page/table provenance اضافه شد؛ علت، به‌هم‌ریختن ترتیب ستون‌های جدول در استخراج متن PDF است. این رکوردها ادعای بیرونی ندارند.
4. `all-minilm` برای هر رکورد بردار ۳۸۴بعدی تولید می‌کند.
5. بردارها L2-normalize و در `FAISS IndexFlatIP` ذخیره می‌شوند؛ inner product بردارهای نرمال‌شده همان cosine similarity است.
6. سه chunk برتر به Qwen2.5-1.5B داده می‌شوند. Citationهای تولیدی اعتبارسنجی می‌شوند تا مدل شماره صفحه‌ای خارج از retrieval نسازد.
7. اگر بالاترین cosine کمتر از ۰.۴۷ باشد، برنامه ابتدا اعلام می‌کند پاسخ در مقالات نیست و بعد پاسخ عمومی مدل محلی را جدا می‌آورد.

![معماری](outputs/architecture.png)

## توابع

سه تابع اجباری:

- `extract_experimental_results()`
- `compare_results()`؛ جدول CSV و نمودار PNG می‌سازد.
- `generate_survey()`

سه تابع امتیازی:

- `find_research_gap()`
- `compare_references()`؛ شمارش heuristic سال‌های References همراه با توضیح محدودیت.
- `compare_limitations()`

مدل ابتدا با schema رسمی ابزارهای Ollama تصمیم می‌گیرد. اگر مدل ۱.۵B tool call معتبر نسازد، یک fallback واژه‌محور اعتبارسنجی‌شده وجود دارد تا CLI خراب نشود. در آزمون ۱۲ درخواست فارسی/انگلیسی، ۱۱ مسیر مستقیماً با LLM tool call و یک مسیر با fallback انجام شد.

## نصب و اجرا در Windows PowerShell

پیش‌نیازها: Python 3.11 و Ollama.

پیش از اجرای setup، چهار PDF مقاله را طبق [راهنمای آماده‌سازی مقاله‌ها](Researchs/README.md) در پوشهٔ `Researchs` قرار دهید.

```powershell
.\setup.ps1
```

اجرای CLI:

```powershell
.\run_demo.ps1
```

اجرای یک پرسش:

```powershell
.\.venv\Scripts\python cli.py -q "What Natural Questions test Exact Match is reported for Single DPR?"
```

رابط Streamlit اختیاری:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

بازسازی index و ارزیابی:

```powershell
.\.venv\Scripts\python scripts\build_index.py
.\.venv\Scripts\python scripts\run_evaluation.py
.\.venv\Scripts\python scripts\generate_demo_outputs.py
.\.venv\Scripts\python -m pytest -q
```

## پرسش‌های مناسب برای دمو

```text
What Natural Questions test Exact Match is reported for Single DPR?
How does Fusion-in-Decoder combine evidence from multiple passages?
Compare the results of all papers and draw a chart.
Generate a mini survey of the selected papers.
Find research gaps and future work.
Compare references by publication year.
What is the weather in Tehran tomorrow?
```

پرسش آخر عمداً خارج از دامنه است و باید قالب fallback الزامی را نشان دهد.

## نتایج واقعی همین اجرا

محیط آزمایش: Windows 11 Pro، پردازنده‌ی Intel Core i7-11800H، رم ۱۵.۸ گیگابایت، GPU لپ‌تاپی RTX 3060، Python 3.11.9 و Ollama 0.15.6.

| ارزیابی | نتیجه |
|---|---:|
| Top-1 paper retrieval accuracy | 85.00% |
| Retrieval Hit@3 | 100.00% |
| Retrieval Precision@3 | 81.67% |
| Retrieval Recall@3 | 96.25% |
| Retrieval F1@3 | 85.12% |
| Tool-routing accuracy (12 query) | 100.00% |
| No-RAG answer containment (12 query) | 8.33% |
| RAG answer containment (12 query) | 75.00% |
| No-RAG mean Token F1 | 9.72% |
| RAG mean Token F1 | 70.83% |

OOD threshold با ۱۵ نمونه‌ی calibration انتخاب و روی ۱۵ نمونه‌ی held-out آزمایش شد؛ نتیجه در این مجموعه‌ی کوچک ۱۰۰٪ بود، اما این عدد ادعای تعمیم به همه پرسش‌های دنیا نیست.

PCA دوبعدی فقط ۱۱.۸۸٪ واریانس را توضیح داد. بهترین K-Means برابر ۲ بود، اما Silhouette فقط ۰.۰۶۳۸ و purity مقاله فقط ۰.۳۳۸۱ شد. این نتیجه‌ی ضعیف حذف نشده است: chunkها بیشتر بر اساس موضوعات مشترک RAG هم‌پوشانی دارند و خوشه‌بندی برای retrieval اصلی استفاده نمی‌شود.

## محدودیت‌های صادقانه

- مدل ۱.۵B در سه پرسش از ۱۲ پرسش تولیدی جواب درست نداد؛ از جمله optimizer مربوط به FiD و نام FAISS. این همان trade-off مدل سبک است.
- PDFهای دو ستونه و جدول‌ها در استخراج خطی به‌هم می‌ریزند؛ رکوردهای human-verified فقط برای مهم‌ترین جدول مشترک اضافه شدند.
- مجموعه ارزیابی کوچک و دست‌ساز است و confidence interval یا چند seed ندارد.
- EM و answer-containment پاسخ‌های معنایی را کامل نمی‌سنجند.
- مقایسه‌ی EM مقاله‌ها controlled experiment نیست.
- شمارش سال منابع Regex-based و تقریبی است.
- Streamlit و نمودارها امتیازی‌اند؛ هسته‌ی لازم پروژه CLI است.

نمونهٔ خروجی‌ها در [demo_outputs.md](outputs/demo_outputs.md) قرار دارد.
