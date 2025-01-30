import subprocess

print("🚀 Запускаем обновление базы лиц...")
subprocess.run(["python3", "encode_faces.py"], check=True)

print("🎥 Запускаем основную программу...")
subprocess.run(["python3", "main.py"], check=True)
