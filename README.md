# 🧚 맞춤형 AI 동화책 생성기 (Google Gemini & Imagen 2)

이 프로젝트는 Google Gemini 1.5 Pro 및 Flash 모델을 사용하여 맞춤형 동화책을 생성하고, Google Imagen 2를 활용하여 각 챕터에 대한 이미지를 생성하는 Streamlit 애플리케이션입니다.

## 주요 기능

- 사용자 지정 주제 및 장르에 따른 동화책 생성
- 다국어 지원 (한국어, 영어, 일본어, 독일어, 중국어, 프랑스어, 스페인어, 베트남어)
- 챕터 수와 챕터당 단어 수 설정 가능
- AI 생성 이미지를 각 챕터에 포함
- 생성된 동화를 Markdown 및 PDF 형식으로 다운로드 가능

## 설치 방법

1. 저장소를 클론합니다:
   ```
   git clone https://github.com/maisonde3cochons/gemini1.5-children-storybook.git
   cd gemini1.5-children-storybook
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. `.env` 파일을 생성하고 필요한 API 키를 설정합니다:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. Google Cloud 서비스 계정 키 파일을 준비하고 경로를 설정합니다.

## 사용 방법

1. Streamlit 앱을 실행합니다:
   ```
   python3 -m streamlit run gemini-long-text-imagen2.py
   ```

2. 웹 브라우저에서 표시된 URL로 이동합니다.

3. 언어, 동화책 주제, 장르, 챕터 수, 챕터당 단어 수를 선택합니다.

4. "동화책 생성하기" 버튼을 클릭하고 생성이 완료될 때까지 기다립니다.(길이에 따라서 약 3~8분 소요)

5. 생성된 동화책을 화면에서 확인하고, Markdown 또는 PDF 형식으로 다운로드할 수 있습니다.

## 주의사항

- Google Cloud 서비스 사용을 위해 적절한 권한과 설정이 필요합니다.
- API 사용량과 관련된 비용에 주의하세요.

## 기여 방법

프로젝트 개선에 기여하고 싶으시다면 언제든 Pull Request를 보내주세요. 큰 변경사항의 경우, 먼저 Issue를 열어 논의해주시기 바랍니다.

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)에 따라 라이선스가 부여됩니다.
