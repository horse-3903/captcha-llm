import os
import pyfiglet
import random
import string

fonts = [
    'charact1', 'charact2', 'charact3', 'charact4', 'charact5', 'charact6', 'characte',
    'univers', 'nancyj-fancy', 'nancyj-underlined', 'nancyj-improved',
    'amc_aaa01', 'amc_slash', 'ascii___', 'bubble__', 'char2___',
    'com_sen_', 'cosmic', 'cosmike', 'demo_1__', 'demo_m__', 'doh',
    'ebbs_1__', 'ebbs_2__', 'e__fist_', 'fbr2____', 'filter',
    'fraktur', 'georgi16', 'georgia11', 'gothic__', 'new_asci',
    'nscript', 'nvscript', 'o8', 'radical_', 'roman', 'roman___',
    'space_op', 't__of_ap', 'ucf_fan_', 'utopiab', 'utopiabi',
    'xhelvb', 'xhelvbi', 'xsansbi', 'xsbookb', 'xsbookbi', 'xtimes', 'xttyb'
]

def generate_captcha(length: int):
    char_set = string.ascii_uppercase + string.digits
    captcha_text = ''.join(random.choices(char_set, k=length))
    return captcha_text

def format_captcha(captcha: str, font: str):
    assert font in fonts
    res = pyfiglet.figlet_format(captcha, font=font)
    return res

def main(num_samples=500, captcha_length=None):
    out_dir = os.path.join('data', 'ascii-captcha')
    os.makedirs(out_dir, exist_ok=True)
    
    for _ in range(num_samples):
        if not captcha_length:
            captcha_length = random.randint(7, 15)

        captcha = generate_captcha(captcha_length)
        font = random.choice(fonts)
        ascii_art = format_captcha(captcha, font)
        
        file_path = os.path.join(out_dir, f"{captcha}.txt")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(ascii_art)

if __name__ == "__main__":
    main()