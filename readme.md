# 将pdf解析为markdown并保存pdf中的图像、题注

使用VLM以及语言模型对pdf进行解析（如果gpu资源不够，可均使用VLM进行解析）

1、使用 conda env create -f environment.yml 根据导出的包名文件安装环境

2、配置config文件

"vlm_api_key": "EMPTY",          VLM的apikey，如用vllm启动的模型，设置为EMPTY
"vlm_api_url": "",               VLM的url，vllm启动的根据主机的ip及端口确定
"markdown_api_key": "",          语言模型的apikey，将html转化为md，如gpu资源不够与VLM设置相同apikey
"markdown_api_url": "",          语言模型的url，将html转化为md，如gpu资源不够与VLM设置相同apikey
"input_pdf": "./path_to_input/input_pdf",            待解析文件夹的文件路径
"output_root": "./path_to_output/output"              解析完内容文件夹的总输出路径

输出文件夹结构为：
output/
├── pdf_name_1/
│   ├── figures/        保存pdf页面中的图像内容
│   ├── pdf_pages/      将pdf逐页转换为图片以供解析
│   ├── pdf_pages_bbox/  包含bbox的pdf页面 
│   ├── pdf_name.md
│   └── captions.json
├── pdf_name_2/
└── pdf_name_3/

3、运行
python pdf_extraction.py --config ./path_to_config/config.json     修改为config.json存放路径




