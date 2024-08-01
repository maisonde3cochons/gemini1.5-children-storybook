import streamlit as st
import os
import base64
import io, json
from langchain_core.messages import AIMessage, HumanMessage

from google.oauth2 import service_account
from google.cloud import aiplatform

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from crewai_tools.tools import FileReadTool
from weasyprint import HTML, CSS
from markdown import markdown
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat

from PIL import Image

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="동화책 생성기", page_icon="📚")
st.title("🧚 맞춤형 AI 동화책 생성기 (Google Gemini & Imagen2)")


# 환경 변수에서 Gemini API 키 가져오기
gemini_api_key = os.getenv('GEMINI_API_KEY')

if not gemini_api_key:
    st.error("Gemini API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
else:

    # Pro
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=gemini_api_key,
        verbose=True,
        temperature=0.6,
    )

    # Flash
    flash = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=gemini_api_key,
        verbose=True,
        temperature=0.6,
    )


    # 언어 선택 옵션 추가
    languages = {
        "한국어": "ko",
        "English": "en",
        "日本語": "ja",
        "Deutsch": "de",
        "中文": "zh",
        "Français": "fr",
        "Español": "es",
        "Tiếng Việt": "vi"
    }

    # User inputs
    selected_language = st.selectbox("언어를 선택해주세요", list(languages.keys()))
    language_code = languages[selected_language]    
    story_theme = st.text_input("동화책의 주제를 입력해주세요 (예: 동물에 관한)", "동물에 관한")
    story_genre = st.selectbox("동화책의 장르를 선택해주세요", 
                            ["모험", "교육", "공상과학", "미스터리", "동물 이야기", "우정", "가족"])
    num_chapters = st.number_input("챕터 수를 입력해주세요", min_value=1, max_value=8, value=3)
    words_per_chapter = st.number_input("챕터당 단어 수를 입력해주세요", min_value=30, max_value=350, value=200)

    # Tool definitions
    file_read_tool = FileReadTool(
        file_path='template.md',
        description='A tool to read the Story Template file and understand the expected output format.'
    )


    def load_service_account_key(key_path):
        try:
            with open(key_path, 'r') as key_file:
                key_data = json.load(key_file)
            
            required_fields = ['client_email', 'token_uri', 'private_key', 'project_id']
            for field in required_fields:
                if field not in key_data:
                    raise ValueError(f"Service account key is missing required field: {field}")
            
            return key_data
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in service account key file: {key_path}")
        except IOError:
            raise ValueError(f"Unable to read service account key file: {key_path}")
        

    @tool
    def generateimage(chapter_content_and_character_details: str) -> str:
        """    
        Generates an image for a given chapter content and character details.
        chapter_content_and_character_details must be in english.
        Using the Google Imagen on Vertex AI,
        saves it in the current folder, and returns the image path.
        """

        # 서비스 계정 키 파일 경로 (환경 변수에서 가져오거나 직접 지정)
        key_path = "/engn001/gcp-api-cred/proven-yen-430302-n2-abe9c79cbabf.json"
        if not key_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
        
        # 서비스 계정 키 로드 및 검증
        key_data = load_service_account_key(key_path)
        
        # 서비스 계정 인증 정보 생성
        try:
            credentials = service_account.Credentials.from_service_account_info(
                key_data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except ValueError as e:
            raise ValueError(f"Error creating credentials: {str(e)}")
        
        # Vertex AI 초기화
        project_id = key_data['project_id']
        try:
            aiplatform.init(project=project_id, location='us-central1', credentials=credentials)
        except Exception as e:
            raise ValueError(f"Error initializing AI Platform: {str(e)}")

        # 이미지 생성 모델 객체 생성
        generator = VertexAIImageGeneratorChat()

        # 프롬프트 준비
        prompt = f'''Image is about: {chapter_content_and_character_details}. Style: Illustration. Create an illustration incorporating a vivid palette with an emphasis on shades of azure and emerald, augmented by splashes of gold for contrast and visual interest. The style should evoke the intricate detail and whimsy of early 20th-century storybook illustrations, blending realism with fantastical elements to create a sense of wonder and enchantment. The composition should be rich in texture, with a soft, luminous lighting that enhances the magical atmosphere. Attention to the interplay of light and shadow will add depth and dimensionality, inviting the viewer to delve into the scene. DON'T include ANY text in this image. DON'T include colour palettes in this image.'''
#        prompt = f"Image is about: {chapter_content_and_character_details}. Create an illustration incorporating a vivid palette with an emphasis on shades of azure and emerald, augmented by splashes of gold for contrast and visual interest. Style : children storybook illustrations. DON'T include ANY text in this image. DON'T include colour palettes in this image"
#        prompt = f"Image is about: {chapter_content_and_character_details}. Children storybook illustrations in watercolor style."

        # Negative prompt 추가
        negative_prompt = f'''text in this image. 
        colour palettes in this image. 
        generate distorted or extra body parts. 
        create inappropriate or adult content. 
        creating blurry or low-quality images.'''

        # 메시지 생성 및 이미지 생성
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "text", "text": f"Negative prompt: {negative_prompt}"}
            ])
        ] 

        # 메시지 생성 및 이미지 생성
        messages = [HumanMessage(content=[prompt])]
        try:
            response = generator.invoke(messages)
        except Exception as e:
            raise ValueError(f"Error generating image: {str(e)}")

        # 응답 객체에서 base64 문자열 파싱
        generated_image = response.content[0]
        img_base64 = generated_image["image_url"]["url"].split(",")[-1]

        # base64 문자열을 이미지로 변환
        img = Image.open(io.BytesIO(base64.b64decode(img_base64)))

        # 프롬프트의 첫 몇 단어를 기반으로 파일 이름 생성
        words = chapter_content_and_character_details.split()[:5]
        safe_words = []
        for word in words:
            safe_word = ''.join(c for c in word if c.isalnum() or c in ['-', '_'])
            if safe_word:  # 빈 문자열이 아닌 경우에만 추가
                safe_words.append(safe_word)
        
        if not safe_words:  # 모든 단어가 제거된 경우
            safe_words = ['image']  # 기본 파일 이름 사용
        
        filename = "_".join(safe_words).lower() + ".png"
        
        # 파일 이름이 비어있지 않은지 확인
        if filename == ".png":
            filename = "generated_image.png"
        
        filepath = os.path.join(os.getcwd(), filename)

        # 이미지 저장
        img.save(filepath)
        print(f"Image saved at: {filepath}")

        return filepath



    @tool
    def convert_markdown_to_pdf(markdownfile_name: str) -> str:
        """Converts a Markdown file to a PDF document using WeasyPrint."""
        output_file = os.path.splitext(markdownfile_name)[0] + '.pdf'
        
        # Read the Markdown content
        with open(markdownfile_name, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        
        # Convert Markdown to HTML
        html_content = markdown(markdown_content)

        # Add CSS for basic styling and to handle Korean font
        css_content = """
        @font-face {
            font-family: 'NanumGothic';
            src: url('https://fonts.googleapis.com/css2?family=Nanum+Gothic&display=swap');
        }
        body {
            font-family: 'NanumGothic', sans-serif;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        """

        # Function to handle image paths
        def image_uri(uri):
            if uri.startswith('http'):
                return uri
            else:
                with open(uri, 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded_image}"

        # Create PDF
        HTML(string=html_content, base_url=os.getcwd()).write_pdf(
            output_file,
            stylesheets=[CSS(string=css_content)],
            presentational_hints=True,
            optimize_images=True,
            font_config=None,
            image_handler=image_uri
        )

        return output_file

    # Agent definitions
    story_outliner = Agent(
        role='Story Outliner',
        goal=f'{story_theme} 이야기. {story_genre} 장르의 어린이 동화책 개요를 작성하고, {num_chapters}개의 챕터에 대한 챕터 제목과 등장인물을 포함하세요. {selected_language}로 작성해줘',
        backstory="어린이를 위한 재미있는 이야기 개요를 만드는 상상력 넘치는 크리에이터입니다.",
        verbose=True,
        llm=flash,
        allow_delegation=False
    )

    story_writer = Agent(
        role='Story Writer',
        goal=f'각 장마다 약 {words_per_chapter}단어씩 총 {num_chapters}개의 챕터에 대한 이야기의 전체 내용을 작성하고, 윤곽이 잡힌 이야기와 등장인물을 엮어서 작성합니다. {selected_language}로 작성해줘',
        backstory="동화 세계와 캐릭터에 생명을 불어넣어 어린이를 위한 흥미롭고 상상력이 풍부한 이야기를 만들어내는 재능 있는 스토리텔러입니다.",
        verbose=True,
        llm=flash,
        allow_delegation=False
    )

    image_generator = Agent(
        role='Image Generator',
        goal=f'''Create one image for each chapter content provided by the story outline. Create a one-sentence (20 words or less) description of the chapter title and supporting characters in english and pass it as Input. Generate one image for a total of {num_chapters} images. The final output should contain all {num_chapters} images in json format. chapter_content_and_character_details must be in English''',
        backstory="A creative AI that specialises in visual storytelling, bringing each chapter to life through imaginative imagery.",
        verbose=True,
        llm=llm,
        tools=[generateimage],
        allow_delegation=False
    )

    content_formatter = Agent(
        role='Content Formatter',
        goal='''각 장의 시작 부분에 "image_generator" Agent가 생성한 이미지를 포함하여 아래와 같은 Format의 마크다운 형식으로 만들어줘. Output Markdown File 내에는 설명을 넣지마. : 
            # Title of the Book

            ## Chapter 1: The Magic Forest
            ![Chapter 1 Image](/engn001/generated_image1.png)
            Contents of chapter 1...

            ## Chapter 2: The Hidden Cave
            ![Chapter 2 Image](/engn001/generated_image2.png)
            Contents of chapter 2...

            ## Chapter 3: The Old Wise Tree
            ![Chapter 3 Image](/engn001/generated_image3.png)
            Contents of chapter 3...

            ## Chapter 4: The River of Dreams
            ![Chapter 4 Image](/engn001/generated_image4.png)
            Contents of chapter 4...

            ## Chapter 5: The Return Journey
            ![Chapter 5 Image](/engn001/generated_image5.png)
            Contents of chapter 5...
                
        ''',
        backstory='스토리북의 가독성과 표현력을 높여주는 꼼꼼한 포맷터입니다.',
        verbose=True,
        llm=llm,
        # tools=[file_read_tool],
        allow_delegation=False
    )

    markdown_to_pdf_creator = Agent(
        role='PDF Converter',
        goal='Convert the Markdown file to a PDF document. story.md is the markdown file name.',
        backstory='An efficient converter that transforms Markdown files into professionally formatted PDF documents.',
        verbose=True,
        llm=flash,
        tools=[convert_markdown_to_pdf],
        allow_delegation=False
    )

    # Task definitions
    task_outline = Task(
        description=f'{story_theme} 관련 이야기. {story_genre} 장르의 어린이 동화책 개요를 작성하고 {num_chapters}개의 챕터에 대한 챕터 제목과 캐릭터 설명을 자세히 설명합니다. {selected_language}로 작성해주세요.',
        agent=story_outliner,
        expected_output=f'{num_chapters}개의 챕터 제목이 포함된 구조화된 개요 문서로, 각 챕터의 자세한 캐릭터 설명과 주요 줄거리가 포함되어 있습니다.'
    )

    task_write = Task(
        description=f'제공된 개요를 사용하여 모든 챕터의 전체 이야기 내용을 작성하여 어린이가 일관성 있고 흥미를 가질 수 있는 이야기를 구성하세요. 각 장은 {words_per_chapter}단어로 작성합니다. 상단에 이야기 제목을 포함하세요.  {selected_language}로 작성해주세요.',
        agent=story_writer,
        expected_output=f'{story_theme} 이야기. {story_genre} 장르의 어린이 동화책의 전체 원고를 {num_chapters}개의 챕터로 구성하세요. 각 장은 제공된 개요를 따르고 등장인물과 줄거리를 일관된 이야기로 통합하여 약 {words_per_chapter}단어를 포함해야 합니다.'
    )

    task_image_generate = Task(
        description=f'''{story_theme} stories. {story_genre} Create {num_chapters} images that capture the essence of a children's storybook in your genre, outlining the theme, characters, and chapters. Create them one by one.''',
        agent=image_generator,
        context=[task_outline],
        expected_output=f'''A digital image file that is a visual representation of an important theme in a children's storybook, incorporating elements of character and plot as described in the brief. The image should be suitable for inclusion as an illustration in a children's book.'''
    )


    task_format_content = Task(
        description='각 장의 시작 부분에 이미지를 포함하여 스토리 콘텐츠의 형식을 마크다운으로 지정합니다.',
        agent=content_formatter,
        expected_output='전체 스토리북 콘텐츠는 각 챕터 제목 뒤에 해당 이미지와 챕터 콘텐츠가 마크다운 형식으로 표시됩니다.',
        context=[task_write, task_image_generate],
        output_file="story.md"
    )

    task_markdown_to_pdf = Task(
        description='Convert a Markdown file to a PDF document, ensuring the preservation of formatting, structure, and embedded images using the mdpdf library.',
        agent=markdown_to_pdf_creator,
        expected_output='A PDF file generated from the Markdown input, accurately reflecting the content with proper formatting. The PDF should be ready for sharing or printing.'
    )

    # Crew setup
    crew = Crew(
        agents=[story_outliner, story_writer, image_generator, content_formatter, markdown_to_pdf_creator],
        tasks=[task_outline, task_write, task_image_generate, task_format_content, task_markdown_to_pdf],
        verbose=True,
        process=Process.sequential
    )

    # Streamlit UI
    st.write("이 앱은 Google Gemini1.5 Pro 및 Flash 모델을 사용하여 동화책을 생성하고, Imagen 2를 사용하여 이미지를 생성합니다.")

    if st.button("동화책 생성하기"):
        with st.spinner(f"'{story_theme}' 주제의 {story_genre} 장르 동화책을 생성하는 중입니다... 잠시만 기다려주세요."):
            result = crew.kickoff()
        
        st.success("동화책 생성이 완료되었습니다!")
        
        # Display the generated story
        with open("story.md", "r", encoding="utf-8") as file:
            story_content = file.read()
        st.markdown(story_content)
        
        # Function to create a download link
        def get_download_link(file_path, link_text):
            with open(file_path, "rb") as file:
                contents = file.read()
            b64 = base64.b64encode(contents).decode()
            return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path}" target="_blank">{link_text}</a>'
        
        # Markdown file download link
        st.markdown(get_download_link("story.md", "Markdown 파일 다운로드"), unsafe_allow_html=True)
        
        # PDF file download link
        st.markdown(get_download_link("story.pdf", "PDF 파일 다운로드"), unsafe_allow_html=True)