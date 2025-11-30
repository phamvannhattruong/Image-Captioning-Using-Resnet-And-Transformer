import google.generativeai as genai
import os

# --- CẤU HÌNH ---
# Thay 'YOUR_API_KEY_HERE' bằng key bạn vừa lấy được
# Hoặc tốt hơn là lưu trong biến môi trường để bảo mật
API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=API_KEY)


class GeminiCaptionRefiner:
    def __init__(self):
        # Sử dụng gemini-1.5-flash vì nó nhanh (low latency) và miễn phí
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def refine(self, raw_caption):
        # Prompt hướng dẫn model sửa lỗi nhưng cấm "chém gió" thêm
        prompt = (
            f"You are a helpful assistant correcting image captions.\n"
            f"Input raw caption: '{raw_caption}'\n"
            f"Task: Fix grammar errors, remove repetitions, and make the sentence natural.\n"
            f"CRITICAL: Do NOT add new objects, colors, or context not mentioned in the input.\n"
            f"Output: Just the corrected sentence."
        )

        try:
            # Gọi API
            response = self.model.generate_content(prompt)

            # Lấy kết quả text
            return response.text.strip()
        except Exception as e:
            print(f"Lỗi khi gọi Gemini: {e}")
            # Nếu lỗi mạng, trả về luôn caption gốc để không làm crash chương trình
            return raw_caption


# --- CHẠY THỬ ---
if __name__ == "__main__":
    refiner = GeminiCaptionRefiner()

    # Giả lập một số caption lỗi thường gặp
    test_captions = [
        "ateboarder in a red shirt",  # Lỗi ngữ pháp
        "a man holding a holding a cup",  # Lỗi lặp từ (stuttering)
        "cat sleep sofa"  # Câu cụt lủn
    ]

    print("--- KẾT QUẢ SỬA LỖI VỚI GEMINI ---")
    for raw in test_captions:
        refined = refiner.refine(raw)
        print(f"Raw:     {raw}")
        print(f"Refined: {refined}")
        print("-" * 30)