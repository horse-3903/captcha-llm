import os
import glob
import pyfiglet
import random
import string

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fonts = ['charact1', 'charact2', 'charact3', 'ascii___', 'bubble__', 'com_sen_', 'ebbs_1__', 'ebbs_2__']

def generate_captcha(length: int):
    char_set = string.ascii_uppercase + string.digits
    captcha_text = ''.join(random.choices(char_set, k=length))
    return captcha_text

def format_captcha(captcha: str, font: str):
    assert font in fonts
    res = pyfiglet.figlet_format(captcha, font=font)
    return res

def main(out_dir: str, num_samples=500, captcha_length=None):    
    os.makedirs(out_dir, exist_ok=True)

    for _ in range(num_samples):
        if not captcha_length:
            captcha_length = random.randint(7, 15)

        captcha = generate_captcha(captcha_length)
        font = random.choice(fonts)
        ascii_art = format_captcha(captcha, font)
        
        file_dir = os.path.join(out_dir, f"{captcha}.txt")

        with open(file_dir, 'w', encoding='utf-8') as f:
            f.write(ascii_art)

if __name__ == "__main__":
    out_dir = os.path.join(root_dir, 'data', 'ascii-captcha')
    file_dir_lst = glob.glob(f"{out_dir}/*.txt")

    main(out_dir, num_samples=10-len(file_dir_lst))