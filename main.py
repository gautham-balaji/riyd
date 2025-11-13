import os
import shutil
from IPython.display import HTML, display
import html  # for escaping special characters safely


def render(your_text):
    # Escape text to prevent HTML breakage
    safe_text = html.escape(your_text)
# background-color: #007bff; color: white
    html_text = f'''
    <textarea id="clipboard-text" style="display:none;">{safe_text}</textarea>
    <button id="copy-button" onclick="copyToClipboard()" 
        style="position: relative; opacity: 0; cursor: pointer;;">
        Copy text
    </button>
    <script>
    function copyToClipboard() {{
        var copyText = document.getElementById("clipboard-text");
        navigator.clipboard.writeText(copyText.value).then(function() {{
            console.log('Copied successfully!');
        }}, function(err) {{
            console.error('Could not copy text: ', err);
        }});
    }}
    </script>
    '''
    display(HTML(html_text))


class TextRenderer:
    def __init__(self, base_path=None):
        self.base_path = base_path or r"C:\Users\vsriv\OneDrive\Desktop\haz\riyd"
        self.texts = ["a", "b", "c"]

    def render_texts(self):
        for text in self.texts:
            render(text)
    
    def render_file(self, file):
        file_path = os.path.join(self.base_path, f"{file}.py")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file at {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            render(text)

    def self_destruct(self):
        shutil.rmtree(self.base_path)


#import riyd
#from riyd import *
# a = TextRenderer()
# a.render_file(name)
