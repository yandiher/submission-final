# Submission Pertama: Menyelesaikan Permasalahan Human Resources

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan bisnis
Sejauh ini total murid yang dropout lebih dari 25%. Hal ini menimbulkan pertanyaan besar mengapa bisa sebanyak itu murid gagal di tengah jalan untuk melanjutkan pendidikannya.

### Cakupan Proyek

Cakupan proyek ini adalah untuk melihat variabel-variabel apa saja yang berpengaruh secara signifikan terhadap murid yang dropout. lalu melakukan segmentasi murid seperti terbagi dalam menjadi beberapa cluster dilihat dari displaced, debtor, dan tuition fee. output dari proyek ini adalah dashboard visualisasi, model prediction, dan actionable insight yang bisa diterapkan oleh team human resource.

### Persiapan

Sumber data: [link](https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv)

### Setup Environment

~~~bash
pip install virtualenv
virtualenv --version

python -m venv env
cd env/Scripts
activate

# kembali ke folder yang berisi requirements.txt

pip install -r requirements.txt

python -m prediction.py # jika ingin menggunakan machine learning
~~~

## Menjalankan Machine Learning

machine learning dibuat menggunakan algoritma extreme gradient boosting yang dibangun di streamlit. untuk menggunakannya, bisa langsung [klik di sini.](https://submission-final-ds-dicoding.streamlit.app/)

## Business Dashboard

setelah melakukan inferensial statistika, didapatkan dengan menggunakan student ttest dan chi square variabel bahwa variabel-variabel berikut memiliki hubungan yang signifikan dalam menjawab apakah seorang murid dengan kategori tersebut akan dropout atau graduate:
- marital_status
- application_mode
- course
- time_attendance
- previous_qualification
- mother_qualification
- father_qualification
- mother_occupation
- father_occupation
- displaced
- debtor
- tuition_fees_up_to_date
- gender
- scholarship_holder
- application_order
- previous_qualification_grade
- admission_grade
- age_at_enrollment
- Curricular_units_1st_sem_credited
- Curricular_units_1st_sem_enrolled
- Curricular_units_1st_sem_evaluations
- Curricular_units_1st_sem_approved
- Curricular_units_1st_sem_grade
- Curricular_units_1st_sem_without_evaluations
- GDP

Variabel-variabel tersebut diseleksi dan dipilih 10 variabel terbaik untuk menjelaskan kenapa murid bisa dropout atau graduate.

berikut adalah link [dashboard tableau.](https://public.tableau.com/app/profile/yandi.hermawan/viz/DropoutStudentDashboard/Dashboard1)

## Conclution
sebanyak 51% murid dropout kelas malam, 48% murid yang tidak dapat beasiswa dropout, dan sebesar 76% murid dropout yang biaya kuliahnya meminjam, serta 94% murid dropout karena tidak tuition up to date. artinya permasalahan terbesar kampus adalah murid-murid yang kesulitan biaya.

## Recomendation Action
- memberikan beasiswa untuk murid yang tidak melulu soal prestasi, berikan juga beasiswa kepada murid yang tidak mampu membayar kuliah.
- jika kampus tidak mampu memberikan beasiswa, dibuatkan forum untuk diskusi dan saling membantu untuk mendapatkan beasiswa di luar kampus.
