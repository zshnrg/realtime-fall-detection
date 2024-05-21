import cv2
import requests
import time

# URL dari server Flask
server_url = 'http://127.0.0.1:5000/detect_fall'
bot_token = '9rqKxPXzYNCEsrFrwRhDzIL7UgM3ld45dEF7W7KmmLe'

# Membuka koneksi ke webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Mengambil gambar dari webcam
        ret, frame = cap.read()
        
        if ret:
            # Menampilkan gambar di jendela
            cv2.imshow('Webcam', frame)

            # Menyimpan gambar sementara
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            # Mengirim permintaan POST ke server Flask
            response = requests.post(
                server_url,
                files={'image': image_bytes},
                params={'bot_token': bot_token}
            )

            # Memeriksa respon dari server
            if response.status_code == 200:
                print(response.json())
            else:
                print(f"Failed to send image. Status code: {response.status_code}")

            # Tunda sebelum mengambil gambar berikutnya (misalnya, 1 detik)
            time.sleep(1)

            # Jika tombol 'q' ditekan, keluar dari loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture image")
            break
finally:
    # Menutup koneksi ke webcam
    cap.release()
    cv2.destroyAllWindows()