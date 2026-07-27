import os
import requests

essential = {
    'base': 'http://127.0.0.1:8000',
    'register': '/api/auth/register',
    'login': '/api/auth/login',
    'upload': '/api/resumes/upload'
}

# create a small sample PDF
pdf_bytes = b"%PDF-1.1\n1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>endobj\n4 0 obj<< /Length 44 >>stream\nBT /F1 24 Tf 100 700 Td (Hello PDF) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000061 00000 n \n0000000111 00000 n \n0000000200 00000 n \ntrailer<< /Root 1 0 R /Size 5 >>\nstartxref\n300\n%%EOF"

sample_path = os.path.join(os.path.dirname(__file__), 'sample.pdf')
with open(sample_path, 'wb') as f:
    f.write(pdf_bytes)

# Register user
reg = requests.post(essential['base'] + essential['register'], json={"username":"test_e2e","password":"Password123!"})
print('register status', reg.status_code, reg.text[:500])

# Login
login = requests.post(essential['base'] + essential['login'], json={"username":"test_e2e","password":"Password123!"})
print('login status', login.status_code, login.text[:500])
if login.status_code != 200:
    raise SystemExit('login failed')

token = login.json().get('access_token')
headers = {"Authorization": f"Bearer {token}"}

# Upload PDF
with open(sample_path, 'rb') as f:
    files = {'file': ('sample.pdf', f, 'application/pdf')}
    resp = requests.post(essential['base'] + essential['upload'], headers=headers, files=files)
    print('upload status', resp.status_code)
    print(resp.text)

print('done')
