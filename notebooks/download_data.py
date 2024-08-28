import gdown

def download_file(file_id, output_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_name, quiet=False)

# Example usage:
file_id1 = '1KU0EtfjOTVFyoovF2Z7MguHzZm_R_eGu'
file_id2 = '14ZLT_HqmrQVOfG0qtz-u2PbtLdo7Ia9x'
download_file(file_id1, 'fineweb_post.zip')
download_file(file_id2, 'fineweb_pre.zip')