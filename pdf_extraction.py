import os
import re
import json
import requests
import base64
import fitz
import time
import argparse
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup, Tag
from pathlib import Path
from qwen_vl_utils import smart_resize

# 载入参数
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='Path to config file')
args = parser.parse_args()

with open(args.config, 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

VLM_API_KEY = CONFIG["vlm_api_key"]
VLM_API_URL = CONFIG["vlm_api_url"]
MARKDOWN_API_KEY = CONFIG["markdown_api_key"]
MARKDOWN_API_URL = CONFIG["markdown_api_url"]
INPUT_PDF = CONFIG["input_pdf"]
OUTPUT_ROOT = CONFIG["output_root"]

# API重试机制
def retry_api_call(func, wait_sec=5, max_retries=15, name="API调用"):
    attempt = 1
    while attempt <= max_retries:
        try:
            result = func()
            if result:
                return result
            else:
                print(f"{name} 第 {attempt} 次返回无效结果，等待 {wait_sec}s 后重试...")
        except Exception as e:
            print(f"{name} 第 {attempt} 次发生异常：{e}")
        attempt += 1
        time.sleep(wait_sec)
    print(f"{name} 达到最大重试次数 {max_retries} 次，仍未成功。")
    return None

# 压缩图片尺寸，符合VLM输入大小
def compress_image(img_path, max_width=2500, max_height=3000, quality=70):
    with Image.open(img_path) as image:
        orig_w, orig_h = image.width, image.height
        if orig_w > max_width or orig_h > max_height:
            ratio_w = max_width / orig_w
            ratio_h = max_height / orig_h
            resize_ratio = min(ratio_w, ratio_h, 1.0)  
            new_size = (int(orig_w * resize_ratio), int(orig_h * resize_ratio))

            print(f"压缩 {img_path} 原尺寸: {orig_w}x{orig_h} → 新尺寸: {new_size}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            image.save(img_path, format="JPEG", quality=quality, optimize=True)
        else:
            print(f"无需压缩 {img_path} ({orig_w}x{orig_h})")

# 将pdf逐页转化为图像
def process_pdf(pdf_path, output_dir, zoom=3.0, max_width=2500, max_height=3000):

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = os.path.join(output_dir, f"page_{page_index + 1}.jpeg")
        pix.save(img_path)

        compress_image(img_path,max_width=max_width,max_height=max_height,quality=70)

        file_size = os.path.getsize(img_path) / 1024 / 1024
        print(f"保存第 {page_index + 1} 页: {img_path}({file_size})")
        image_paths.append(img_path)

    return image_paths

# 绘制bounding box并根据html标签获取页面图像
def draw_bbox(image_path, image_save_path, fig_dir, resized_width, resized_height, full_predict, page_num):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    draw = ImageDraw.Draw(image)
    soup = BeautifulSoup(full_predict, 'html.parser')
    elements = soup.find_all(attrs={"data-bbox": True})

    image_blocks = []   # [(bbox, tag)]
    other_blocks = []   # [(bbox, text)]

    for el in elements:
        bbox_str = el['data-bbox']
        # text = element.get_text(strip=True)
        x1, y1, x2, y2 = map(int, bbox_str.split())
        class_list = el.get('class', [])
        text = el.get_text(strip=True)
        if "image" in class_list:
            image_blocks.append(((x1, y1, x2, y2), el))
        elif el.name == 'ol':
            continue
        elif el.name == 'li' and el.parent.name == 'ol':
            other_blocks.append(((x1, y1, x2, y2), text))
        else:
            other_blocks.append(((x1, y1, x2, y2), text))
    # font = ImageFont.truetype("/home/baiyipeng/VLMPDF/NotoSansCJK-Regular.ttc", 20)

    # 保存所有 image_blocks 
    for idx, (bbox, _) in enumerate(image_blocks):
        x1, y1, x2, y2 = bbox
        x1_r = int(x1 / scale_x)
        y1_r = int(y1 / scale_y)
        x2_r = int(x2 / scale_x)
        y2_r = int(y2 / scale_y)
        if x1_r > x2_r:
            x1_r, x2_r = x2_r, x1_r
        if y1_r > y2_r:
            y1_r, y2_r = y2_r, y1_r
        crop_img = image.crop((x1_r, y1_r, x2_r, y2_r))
        fig_name = f"p{page_num}_{idx + 1}.jpg"
        save_path = os.path.join(fig_dir, fig_name)
        crop_img.save(save_path)
        print(f"页面图像保存至: {save_path}")

    # 绘制所有bbox区域
    for bbox, _ in other_blocks + image_blocks:
        x1, y1, x2, y2 = bbox
        x1_r = int(x1 / scale_x)
        y1_r = int(y1 / scale_y)
        x2_r = int(x2 / scale_x)
        y2_r = int(y2 / scale_y)
        if x1_r > x2_r:
            x1_r, x2_r = x2_r, x1_r
        if y1_r > y2_r:
            y1_r, y2_r = y2_r, y1_r
        draw.rectangle([x1_r, y1_r, x2_r, y2_r], outline='red', width=2)
        # draw.text((x1_resized, y2_resized), text, fill='black', font=font)

    # 保存整页bbox图像
    save_bbox_path = os.path.join(image_save_path, f"page_{page_num}_bbox.jpeg")
    image.save(save_bbox_path)
    print(f"第 {page_num} 页bbox结果保存至: {save_bbox_path}")

# 清洗html文本
def clean_and_format_html(full_predict):
    soup = BeautifulSoup(full_predict, 'html.parser')
    
    # Regular expression pattern to match 'color' styles in style attributes
    color_pattern = re.compile(r'\bcolor:[^;]+;?')

    # Find all tags with style attributes and remove 'color' styles
    for tag in soup.find_all(style=True):
        original_style = tag.get('style', '')
        new_style = color_pattern.sub('', original_style)
        if not new_style.strip():
            del tag['style']
        else:
            new_style = new_style.rstrip(';')
            tag['style'] = new_style
            
    # Remove 'data-bbox' and 'data-polygon' attributes from all tags
    for attr in ["data-bbox", "data-polygon"]:
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    classes_to_update = ['formula.machine_printed', 'formula.handwritten']
    # Update specific class names in div tags
    for tag in soup.find_all(class_=True):
        if isinstance(tag, Tag) and 'class' in tag.attrs:
            new_classes = [cls if cls not in classes_to_update else 'formula' for cls in tag.get('class', [])]
            tag['class'] = list(dict.fromkeys(new_classes))  # Deduplicate and update class names

    # Clear contents of divs with specific class names and rename their classes
    for div in soup.find_all('div', class_='image caption'):
        div.clear()
        div['class'] = ['image']

    classes_to_clean = ['music sheet', 'chemical formula', 'chart']
    # Clear contents and remove 'format' attributes of tags with specific class names
    for class_name in classes_to_clean:
        for tag in soup.find_all(class_=class_name):
            if isinstance(tag, Tag):
                tag.clear()
                if 'format' in tag.attrs:
                    del tag['format']

    # Manually build the output string
    output = []
    for child in soup.body.children:
        if isinstance(child, Tag):
            output.append(str(child))
            output.append('\n')  # Add newline after each top-level element
        elif isinstance(child, str) and not child.strip():
            continue  # Ignore whitespace text nodes
    complete_html = f"""```html\n<html><body>\n{" ".join(output)}</body></html>\n```"""
    return complete_html


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 调用VLM对图像进行处理解析，转化为html
def inference(image_path, model_id="qwenvl_32b", min_pixels=512*28*28, max_pixels=2048*28*28):
    sys_prompt="You are a helpful pages extractor."
    user_prompt = ("You are an AI specialized in recognizing and extracting text from images. "
                   "Your mission is to analyze the image document with bbox and generate the result in QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity."
                   "Don't miss any part of the picture")
    base64_image = encode_image(image_path)
    def _call():
        client = OpenAI(
            #If the environment variable is not configured, please replace the following line with the Dashscope API Key: api_key="sk-xxx".
            api_key = VLM_API_KEY,
            base_url= VLM_API_URL,
        )

        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                        # PNG image:  f"data:image/png;base64,{base64_image}"
                        # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                        # WEBP image: f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        completion = client.chat.completions.create(
            model = model_id,
            messages = messages,
            )
        return completion.choices[0].message.content if completion and completion.choices else None
    return retry_api_call(_call, name="VLM图像解析")

# 调用语言模型对html文件进行解析，生成markdown格式文本
def html_to_markdown(html_content, model_id="deepseek-chat"):
    system_prompt = "你是一个熟练的文档转换助手。"
    user_prompt = ("请你将解析得到的html格式文档解析为顺序输出的markdown文档，"
                   "同时过滤掉和正文（正文包括标题、作者，摘要，主要文段、参考文献和附录信息）无关的页眉页脚等信息。"
                   "如果存在图像, HTML 中只有此 `<div class='image'>` 标签代表图像，请在前后相邻上下文段落寻找本图的图名或图像标题,请准确识别此标签"
                   "如果找到图名，为每个图像生成一条 `![图像名称](#)` 形式的 Markdown 标题"
                   "如果图像没有在上下文找到明确图名，请输出为 `![null](#)`，确保所有图像都有对应的 Markdown 标题"
                   "直接输出解析后的markdown文档，不需要任何其他说明。注意，不需要翻译。")
    def _call():
        client = OpenAI(
            api_key = MARKDOWN_API_KEY,
            base_url = MARKDOWN_API_URL,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "text", "text": html_content}
            ]}
        ]

        completion = client.chat.completions.create(model=model_id, messages=messages)
        return completion.choices[0].message.content
    return retry_api_call(_call, name="Markdown 转换")

# 主函数，调用上述函数生成md文件、提取题注保存为json、输出页面图像
def main():

    input_pdf= INPUT_PDF   # pdf路径
    output_root = OUTPUT_ROOT  # 提取结果根目录

    for pdf_file in Path(input_pdf).glob("*.pdf"):
        pdf_path = str(pdf_file)
        pdf_name = pdf_file.stem
        print(f"开始提取文件:{pdf_file.name}" ) 
        output_pdf_dir = os.path.join(output_root, pdf_name)
        md_path = os.path.join(output_pdf_dir, f"{pdf_name}.md")       # pdf输出为md格式

        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            print(f"跳过已处理文件: {pdf_file.name}")
            continue

        imgs_save_dir = os.path.join(output_pdf_dir, "pdf_pages")     # pdf页面转换为图像
        imgs_bbox_save_dir = os.path.join(output_pdf_dir, "pdf_pages_bbox")  # pdf-bbox图片
        caption_dir = os.path.join(output_pdf_dir, "captions.json")   # caption.json
        fig_dir = os.path.join(output_pdf_dir, "figures")
        os.makedirs(imgs_save_dir, exist_ok=True)
        os.makedirs(imgs_bbox_save_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        captions_dict = {}
        all_markdowns = []
        try:
            image_paths = process_pdf(pdf_path, imgs_save_dir)

            for idx, img_path in enumerate(image_paths):
                page_num = idx + 1

                try:
                    print(f"提取第 {page_num} 页")
                    image = Image.open(img_path)
                    min_pixels = 512*28*28
                    max_pixels = 2048*28*28
                    width, height = image.size
                    input_height, input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
                    # VLM推理生成html
                    html_output = inference(img_path ,min_pixels=min_pixels, max_pixels=max_pixels)
                    # 绘制bbox并提取图像内容
                    draw_bbox(img_path, imgs_bbox_save_dir, fig_dir, width, height, html_output, page_num)
                    cleaned_html = clean_and_format_html(html_output)
                    # 模型推理HTML转markdown
                    markdown = html_to_markdown(cleaned_html)
                    all_markdowns.append(markdown)

                    matches = re.findall(r'!\[(.*?)\]\(#\)', markdown)
                    for i, caption_text in enumerate(matches, start=1):
                        key = f"p{page_num}_{i}"
                        captions_dict[key] = caption_text.strip()

                    with open(caption_dir, "w", encoding="utf-8") as f:
                        json.dump(captions_dict, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"第 {page_num} 页处理失败：{e}")
                    continue
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(all_markdowns))
            print(f"处理完成：{pdf_name}")
        except Exception as e:
            print(f"文件 {pdf_name} 处理失败：{e}")
            continue
            
if __name__ == "__main__":
    main()