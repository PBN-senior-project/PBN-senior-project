# ⭐ Group: senior!

## ✅ วิธี clone git
1. สร้าง folder หรือไปที่ folder ที่ต้องการ  
2. คลิกขวา กด "Open in Terminal"  
3. ใส่คำสั่งใน command  

```bash
git clone https://github.com/PBN-senior-project/PBN-senior-project.git
````

---

## ⚠️ แจ้งให้ทราบ

เราจะมี branch กันทั้งหมด **3 กลุ่มใหญ่**

1. **main** : จะเป็น branch หลักที่รวมโค้ดที่สมบูรณ์และเป็นฉบับจริงมาไว้ที่ branch นี้
2. **develop** : จะเป็น branch ที่รวมโค้ดเพื่อนๆทุกคนเปรียบเสมือน main มีเพื่อเอามาเช็คโค้ดก่อนเอาไปที่ branch main เผื่อพัง
3. **\[branchชื่อเพื่อนๆ]** : branch นี้จะเป็น branch สำหรับให้เพื่อนได้ push โค้ดของตัวเองเก็บไว้ของใครของมัน

เพื่อป้องกันไม่ให้โค้ดเกินการเมิร์จกัน – ทุกคนจะมี branch ชื่อของตัวเอง ได้แก่
`ploy`, `mint`, `grace`, `fluke`, `pair`

⚠️ **สำคัญ** : ก่อนจะ push งานขึ้น git แจ้งในกลุ่มก่อนทุกครั้ง
⚠️ **สำคัญ** : ห้ามเพื่อนๆ push code ไปที่ branch `"main"` เด็ดขาด

ก่อนจะอัพงาน ทุกคนต้องเช็ค ว่าอยู่ branch ชื่อตัวเองไหม
ด้วยคำสั่ง

```bash
git branch
```

**Example:**

```
  dev
  fluke
  grace
* main   --> เครื่องหมาย * คือ branch ปัจจุบันที่กำลังทำงาน
  mint
  pair
  ploy
```

---

## ✅ Step 1 - วิธีการอัพงานตัวเองขึ้นไปบน git บน branch ตัวเอง

1. เข้าไปที่ terminal ใน vs code หรือ Intellij หรือ ไปที่ CMD ที่ root folder project

2. ตรวจสอบว่าอยู่ branch ตัวเองรึยัง ⭐สำคัญ

```bash
git branch
git switch <ชื่อ branch ตัวเอง>
```

3. เช็กว่าไฟล์ไหนถูกแก้ไขแล้ว \[ไฟล์ไหนที่ถูกแก้ไข จะมีสีแดง และเขียนว่า modified]

```bash
git status
```

4. เลือกไฟล์ที่ต้องการอัพขึ้น

```bash
git add <ชื่อไฟล์>
# หรือถ้าอยาก add ทุกไฟล์ที่แก้
git add .
```

5. ตรวจเช็กว่าไฟล์ถูก add แล้ว (ไฟล์จะมีสีเขียว และเขียนว่า modified)

```bash
git status
```

**Example:**

```
On branch ploy ---> อยู่ branch ชื่อ ploy แล้ว
Your branch is up to date with 'origin/ploy'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.md   ---> ไฟล์นี้ถูกแก้ไข
```

6. Commit เพื่อบันทึกการเปลี่ยนแปลงใน branch ตัวเอง

```bash
git commit -m "feat: อธิบายสิ่งที่ทำ เช่น เพิ่มหน้า Login"
```

7. Push ขึ้นไปที่ GitHub (ครั้งแรกต้อง set upstream)

```bash
git push -u origin <ชื่อ branch ตัวเอง>
# เช่น
git push -u origin ploy
```

---

## ✅ Step 2 - เอางานบน branch ตัวเอง push ต่อไปที่ dev

❌ พอยดูส่วนนี้คนเดียว

1. กลับมาที่ branch develop

```bash
git checkout dev
```

2. ดึงโค้ดล่าสุดของ develop มาก่อน (กันพัง)

```bash
git pull origin dev
```

3. รวมงานจาก branch ของเพื่อน (ตัวอย่าง: mint)

```bash
git merge --no-ff mint -m "merge: mint → dev"
```

4. ดัน develop ขึ้น GitHub

```bash
git push origin dev
```

ตอนนี้โค้ดจากเพื่อนจะถูกรวมเข้าที่ develop แล้ว

---

## ✅ Step 3 - เอาโค้ดจาก develop ไปที่ main

❌ พอยดูส่วนนี้คนเดียว

1. ตรวจสอบว่า deve สมบูรณ์แล้ว (เพื่อนทุกคน push งานเข้ามาใน develop เรียบร้อย, ทดสอบระบบแล้ว)

2. ไปที่ branch dev

```bash
git checkout dev
git pull origin dev
```

3. ไปที่ branch main

```bash
git checkout main
git pull origin main
```

4. merge develop → main

```bash
git merge --no-ff develop -m "merge: develop → main (release version 1.0)"
```

5. push main ขึ้น GitHub

```bash
git push origin main
```

---

## 🛠 ถ้า git add ผิดไฟล์ แต่ยังไม่ได้ commit

สามารถ “ยกเลิก” ได้ตามนี้

1. ยกเลิกเฉพาะไฟล์ที่ add ไปแล้ว

```bash
git restore --staged <ชื่อไฟล์>
```

2. ยกเลิกทุกไฟล์ที่ถูก add ไปแล้ว

```bash
git restore --staged .
```

3. ถ้าคุณอยากกลับไปไฟล์เดิมทั้งหมด (รวมลบการแก้ไขใน working directory ด้วย ⚠️)

```bash
git restore <ชื่อไฟล์>
# หรือทั้งหมด
git restore .
```

⚠️ ระวัง: คำสั่งนี้จะทำให้ไฟล์หายไปเป็นเวอร์ชันก่อนแก้ (undo code)
ปกติถ้าแค่ “ยกเลิก git add” ก่อน commit ใช้

```bash
git restore --staged <file>
```
---


* **ทุกครั้งที่ branch `develop` ถูกอัพเดต**  เพื่อนๆ **ควร pull** จาก `develop` ลง branch ของตัวเองก่อนทำงานต่อ
* ถ้ามีงานค้างอยู่ → Git จะเตือนว่ามีการแก้ไขไฟล์เดียวกันจากหลายคน (**conflict**) เราจะต้องแก้ไข conflict เอง ไม่อย่างนั้นงานจะทับซ้อนกัน

---

## 🔄 Workflow ที่ควรทำ

1. **อยู่ branch ของตัวเอง** เช่น `grace`

   ```bash
   git checkout grace
   ```

2. **ดึงโค้ดล่าสุดจาก develop มารวมเข้ามาใน branch ตัวเอง**

   ```bash
   git fetch origin
   git merge origin/develop
   ```

   > เพื่อให้ branch ของเราอัพเดตตาม `develop` ตลอด

3. **ถ้ามี conflict** (Git แจ้งไฟล์ที่ทับซ้อน) → เปิดไฟล์มาแก้ไขเอง เช่น:

   ```text
   <<<<<<< HEAD
   // ของเรา
   =======
   // ของ develop
   >>>>>>>
   ```

   → ลบส่วนที่ไม่ต้องการออก แล้ว commit ใหม่

4. **ทำงานต่อ/แก้ไขไฟล์ของตัวเองได้ตามปกติ**

5. **commit + push ไปที่ branch ของตัวเอง**

   ```bash
   git add .
   git commit -m "feat: เพิ่มหน้าใหม่"
   git push origin grace
   ```

---

## ⚠️ ถ้า “ทำงานค้างไว้” แล้วต้อง pull develop

* **กรณีแก้ไฟล์คนละไฟล์กับเพื่อน** → Git จะ merge ให้อัตโนมัติ ไม่มีปัญหา
* **กรณีแก้ไฟล์เดียวกันกับเพื่อน (โดยเฉพาะบรรทัดเดียวกัน)** → จะเกิด **merge conflict** ต้องแก้เอง
* **กรณีอยากเก็บงานที่ยังไม่เสร็จ แต่ต้อง pull ก่อน**
  ใช้ `stash` เก็บงานชั่วคราว

  ```bash
  git stash
  git pull origin develop
  git stash pop
  ```

  → จะได้งานของเรากลับมา พร้อมกับโค้ดใหม่จาก `develop`

---
# PBN Senior Project --- Chest X-Ray Classification

โปรเจกต์สำหรับฝึก (Train) Deep Learning Model เพื่อจำแนกความผิดปกติจากภาพ Chest
X-Ray แบบ Multi-label โดยใช้ Docker เพื่อให้สมาชิกในทีมสามารถรันโปรเจกต์ด้วย
environment เดียวกัน

## Target Findings

โมเดลจำแนก 6 Findings:

-   Infiltration
-   Effusion
-   Atelectasis
-   Nodule
-   Mass
-   Pneumothorax

------------------------------------------------------------------------

## 1. โปรแกรมที่ต้องติดตั้ง

สำหรับเครื่องใหม่ ให้ติดตั้งอย่างน้อย:

1.  **Git**
2.  **Docker Desktop**
3.  **NVIDIA Driver** (เฉพาะกรณีต้องการ Train ด้วย NVIDIA GPU)

> ไม่จำเป็นต้องติดตั้ง Python หรือสร้าง `venv` เพื่อรัน Training ผ่าน Docker เพราะ
> Python และ dependencies จะถูกติดตั้งภายใน Docker image

### ตรวจสอบ Git

เปิด Command Prompt (CMD) หรือ PowerShell:

``` cmd
git --version
```

ถ้าติดตั้งสำเร็จ จะเห็นเวอร์ชันของ Git

### ตรวจสอบ Docker

``` cmd
docker --version
docker compose version
```

------------------------------------------------------------------------

## 2. Clone Repository

เปิด CMD แล้วไปยังโฟลเดอร์ที่ต้องการเก็บโปรเจกต์ เช่น:

``` cmd
cd /d D:\senior-project
```

Clone repository:

``` cmd
git clone https://github.com/PBN-senior-project/PBN-senior-project.git
```

เข้าโปรเจกต์:

``` cmd
cd PBN-senior-project
```

หากต้องการใช้ branch `ploy`:

``` cmd
git checkout ploy
```

ตรวจสอบ branch:

``` cmd
git branch
```

ควรเห็น:

``` text
* ploy
```

------------------------------------------------------------------------

## 3. เตรียม Dataset

โปรเจกต์ใช้ไฟล์ metadata:

``` text
Data_Entry_2017.csv
```

และภาพ Chest X-Ray `.png`

ให้นำ Dataset ไปไว้ในโฟลเดอร์:

``` text
archive/
```

ตัวอย่างโครงสร้าง Dataset:

``` text
PBN-senior-project/
│
├── archive/
│   ├── Data_Entry_2017.csv
│   │
│   ├── images_001/
│   │   └── images/
│   │       ├── 00000001_000.png
│   │       └── ...
│   │
│   ├── images_002/
│   │   └── images/
│   │       └── ...
│   │
│   └── ...
│
├── src/
│   └── train_V7.py
│
├── outputs/
├── models_v7/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

**สำคัญ:** ไม่ควรเปลี่ยนชื่อไฟล์ภาพ เพราะโค้ดจับคู่ชื่อภาพกับคอลัมน์ `Image Index` ใน
`Data_Entry_2017.csv`

------------------------------------------------------------------------

## 4. Path ที่ใช้ภายใน Docker

ใน `docker-compose.yml` มีการ mount:

``` yaml
volumes:
  - ./src:/app/src
  - ./archive:/app/archive
  - ./outputs:/app/outputs
  - ./models_v7:/app/models_v7
```

ดังนั้น Python ที่รันอยู่ **ภายใน Docker** ต้องใช้ path แบบ Docker/Linux เช่น:

``` python
CSV_PATH = "/app/archive/Data_Entry_2017.csv"
ARCHIVE_DIR = "/app/archive"
MODEL_SAVE_DIR = "/app/models_v7"
GRAPH_SAVE_DIR = "/app/outputs/graphs_v7"
```

ห้ามใช้ Windows path เช่น:

``` text
D:\senior-project\...
```

ภายใน `train_V7.py` เมื่อรันผ่าน Docker

------------------------------------------------------------------------

## 5. เปิด Docker Desktop

สามารถเปิดผ่าน CMD:

``` cmd
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

รอจน Docker Engine พร้อม แล้วตรวจสอบ:

``` cmd
docker info
```

หาก `docker info` แสดงข้อมูล Server โดยไม่มี connection error แสดงว่า Docker
พร้อมใช้งาน

------------------------------------------------------------------------

## 6. Build Docker Image --- ครั้งแรก

เข้า root ของโปรเจกต์ก่อน:

``` cmd
cd /d D:\senior-project\PBN-senior-project
```

จากนั้น:

``` cmd
docker compose build
```

ครั้งแรกอาจใช้เวลาสักครู่ เพราะ Docker ต้องดาวน์โหลดและติดตั้ง Python libraries

หากมีการแก้ `requirements.txt` หรือมีปัญหา dependency สามารถ build ใหม่ทั้งหมด:

``` cmd
docker compose build --no-cache
```

------------------------------------------------------------------------

## 7. เริ่ม Training

รัน:

``` cmd
docker compose up
```

Docker Compose จะเรียก:

``` text
python src/train_V7.py
```

ภายใน container `pbn-chestxray-v7`

เมื่อ pipeline ทำงาน ควรเห็น log เช่น:

``` text
⏳ Loading Data...
📁 Images found in folders: ...
✅ Matched with CSV: ...
❌ Missing images: ...

Applying Targeted Oversampling for Minority Classes...
...
Found ... validated image filenames.
```

และเมื่อเริ่ม Train:

``` text
Epoch 1/...
```

------------------------------------------------------------------------

## 8. ทดสอบโค้ดก่อน Train จริง

ถ้าต้องการตรวจสอบว่า pipeline รันได้ครบก่อน ไม่จำเป็นต้อง Train 40 epochs

ใน `src/train_V7.py` เปลี่ยน:

``` python
EPOCHS = 40
```

เป็น:

``` python
EPOCHS = 2
```

เมื่อทดสอบผ่านแล้ว ค่อยเปลี่ยนกลับเป็นค่าที่ต้องการสำหรับ Training จริง

เนื่องจาก `./src` ถูก mount เข้า `/app/src` การแก้ไฟล์ Python ไม่จำเป็นต้อง build
image ใหม่

หยุดรอบเดิมแล้วรันใหม่:

``` cmd
docker compose down
docker compose up
```

------------------------------------------------------------------------

## 9. หยุด Training

หากกำลัง attach อยู่กับ `docker compose up`:

``` text
Ctrl + C
```

จากนั้น:

``` cmd
docker compose down
```

------------------------------------------------------------------------

## 10. ดูสถานะและ Log

ดู container ที่กำลังทำงาน:

``` cmd
docker ps
```

ดู log แบบต่อเนื่อง:

``` cmd
docker logs -f pbn-chestxray-v7
```

ดู container ทั้งที่กำลังทำงานและหยุดแล้ว:

``` cmd
docker ps -a
```

------------------------------------------------------------------------

## 11. เมื่อแก้โค้ด ต้อง Build ใหม่ไหม?

### แก้เฉพาะไฟล์ใน `src/`

เช่น:

``` text
src/train_V7.py
```

**ไม่ต้อง build ใหม่**

ใช้:

``` cmd
docker compose down
docker compose up
```

### แก้ `requirements.txt` หรือ `Dockerfile`

**ต้อง build ใหม่**

``` cmd
docker compose down
docker compose build
docker compose up
```

ถ้าต้องการติดตั้ง dependency ใหม่ทั้งหมด:

``` cmd
docker compose build --no-cache
docker compose up
```

------------------------------------------------------------------------

## 12. Common Problems

### 12.1 Docker Engine ไม่ทำงาน

Error เช่น:

``` text
failed to connect to the docker API
dockerDesktopLinuxEngine
```

เปิด Docker Desktop:

``` cmd
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

แล้ว:

``` cmd
docker info
```

------------------------------------------------------------------------

### 12.2 `ModuleNotFoundError: No module named 'cv2'`

โปรเจกต์ใช้ OpenCV สำหรับ CLAHE / image preprocessing

ตรวจสอบว่า `requirements.txt` มี:

``` text
opencv-python-headless
```

จากนั้น:

``` cmd
docker compose down
docker compose build --no-cache
docker compose up
```

------------------------------------------------------------------------

### 12.3 `KeyError: 'Filename'`

CSV ใช้คอลัมน์ชื่อ:

``` text
Image Index
```

และโปรเจกต์ควรสร้าง `filepath` จาก `Image Index` เพื่อหาไฟล์ภาพใน
`images_001`, `images_002`, ...

Generator ควรใช้แนวทาง:

``` python
x_col="filepath"
directory=None
```

ไม่ควรใช้:

``` python
x_col="Filename"
```

------------------------------------------------------------------------

### 12.4 `Found 0 validated image filenames`

หมายความว่า Keras หาไฟล์ภาพไม่เจอ

ตรวจสอบว่า Dataset อยู่ใต้:

``` text
archive/images_001/images/
archive/images_002/images/
...
```

และ Docker mount:

``` yaml
- ./archive:/app/archive
```

สามารถตรวจสอบไฟล์ภายใน container ได้ เช่น:

``` cmd
docker exec -it pbn-chestxray-v7 sh
```

จากนั้น:

``` sh
find /app/archive -name "00000001_000.png"
```

------------------------------------------------------------------------

### 12.5 CUDA / GPU Warning

อาจพบ:

``` text
Could not find cuda drivers on your machine, GPU will not be used.
```

หรือ:

``` text
failed call to cuInit
```

หมายความว่า TensorFlow ใน container ยังไม่สามารถใช้ CUDA/GPU ได้

หากไม่มี Python traceback และโปรแกรมยังทำงานต่อ โปรแกรมสามารถใช้ CPU ได้ แต่
Training จะช้ากว่า GPU

ตรวจสอบ NVIDIA GPU บน Windows:

``` cmd
nvidia-smi
```

------------------------------------------------------------------------

## 13. Output

Docker mount output กลับมายังเครื่อง:

``` yaml
- ./outputs:/app/outputs
- ./models_v7:/app/models_v7
```

ดังนั้นผลลัพธ์ที่โค้ดบันทึกจะอยู่ใน:

``` text
outputs/
models_v7/
```

โดย `models_v7/` ใช้สำหรับโมเดลที่บันทึกจาก Training และ `outputs/`
ใช้สำหรับผลลัพธ์/กราฟตามที่ Training script สร้าง

------------------------------------------------------------------------

## 14. Quick Start

สำหรับเครื่องที่ติดตั้ง Git + Docker Desktop และเตรียม Dataset เรียบร้อยแล้ว:

``` cmd
git clone https://github.com/PBN-senior-project/PBN-senior-project.git
cd PBN-senior-project
git checkout ploy

docker compose build
docker compose up
```

ครั้งต่อไป หากไม่ได้แก้ `Dockerfile` หรือ `requirements.txt`:

``` cmd
docker compose up
```

------------------------------------------------------------------------

## 15. Quick Command Reference

``` cmd
:: เปิด Docker Desktop
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

:: ตรวจสอบ Docker
docker info

:: Build
docker compose build

:: Run
docker compose up

:: Stop / remove compose containers
docker compose down

:: ดู container
docker ps

:: ดู log
docker logs -f pbn-chestxray-v7

:: Build dependency ใหม่ทั้งหมด
docker compose build --no-cache
```




