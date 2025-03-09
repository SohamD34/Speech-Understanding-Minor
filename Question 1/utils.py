import logging

def setup_logger(log_file_path):
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh.setLevel(logging.INFO)    
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger



def log_text(log_file_path, text):
    logger = setup_logger(log_file_path)
    logger.info(text)