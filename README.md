❤️ FitAI – Smart Fitness & Health Assistant

FitAI là hệ thống trợ lý sức khỏe – dinh dưỡng – luyện tập thông minh, kết hợp:
🧠 RAG (Retrieval-Augmented Generation)
⚡ Qdrant Vector Search
🔥 LLM (Ollama – Qwen 2.5 3B)
🧩 Dynamic User Profiling (tuổi / giới tính / vùng miền / mục tiêu)
📊 BMR/TDEE Prediction Engine
🍱 Meal Planning + Exercise Planning AI

📌 I. Tính năng nổi bật

🔍 1. RAG Search Engine
Nhúng dữ liệu từ 4 dataset Kaggle (80k – 200k rows)
Lưu embedding vào Qdrant Cloud
Tìm kiếm theo ngữ nghĩa (semantic search)

🧠 2. Smart Reasoning Module
Trả lời chuyên sâu theo bối cảnh
Tự động mở rộng suy luận
Gợi ý chi tiết & theo từng nhóm đối tượng

👤 3. Personal Health Profile
Hỗ trợ phân tích theo:
Tuổi
Giới tính
Dân văn phòng / công nhân nặng / học sinh
Mục tiêu (giảm mỡ, tăng cơ, giữ cân)
Khu vực sinh sống (miền Bắc / Trung / Nam → khẩu vị khác nhau)

🔢 4. BMR – TDEE Prediction
Auto nhận diện:
BMI
Mức độ vận động
Ước tính calo duy trì
Tạo meal plan theo target calories

🍽 5. Meal Composer
Tự đề xuất thực đơn theo từng bữa
Gợi ý món Việt Nam (theo vùng miền)
Tính macro, calo, cân đối dưỡng chất

🏋️ 6. Workout Generator
Gợi ý bài tập theo mục tiêu (mỡ bụng, vai – lưng – chân, full body…)
Độ khó: Beginner → Intermediate → Advanced
Có thể tạo “Weekly Training Schedule”

⚙️ II. Cài đặt & Chạy
1. Clone Project
git clone https://github.com/<yourname>/FitAI-Capstone.git
cd FitAI-Capstone/fitai-rag-qdrant

2. Tạo môi trường
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

3. Tạo file .env

Tạo file:
QDRANT_URL=YOUR_QDRANT_URL
QDRANT_API_KEY=YOUR_KEY
EMBEDDING_MODEL=BAAI/bge-m3
OLLAMA_MODEL=qwen2.5:3b-instruct

4. Build index (nhúng dữ liệu)
python -m src.build_index
