import pefile
import os
import numpy as np

def extract_features(file_path, seq_len=4096):
    try:
        with open(file_path, 'rb') as f:
            # 1. 提取指令序列 (强制固定长度)
            raw_bytes = f.read(seq_len)
            bytes_seq = list(raw_bytes)
            if len(bytes_seq) < seq_len:
                bytes_seq += [256] * (seq_len - len(bytes_seq))
            else:
                bytes_seq = bytes_seq[:seq_len]
        
        # 2. 提取 PE 元数据 (强制 10 维)
        pe = pefile.PE(file_path, fast_load=True)
        pe.parse_data_directories()
        
        num_sections = len(pe.sections)
        entry_point = getattr(pe.OPTIONAL_HEADER, 'AddressOfEntryPoint', 0)
        num_imports = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT if entry.imports) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0
        num_exports = len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else 0
        
        # 计算信息熵
        section_entropies = [s.get_entropy() for s in pe.sections]
        avg_entropy = sum(section_entropies) / num_sections if num_sections > 0 else 0
        max_entropy = max(section_entropies) if num_sections > 0 else 0
        
        file_size = os.path.getsize(file_path)
        
        meta = [
            float(num_sections),
            float(entry_point) / 1e6,
            float(num_imports),
            float(num_exports),
            float(avg_entropy),
            float(max_entropy),
            float(file_size) / 1e5,
            float(pe.FILE_HEADER.Machine),
            float(pe.FILE_HEADER.NumberOfSymbols),
            float(getattr(pe.OPTIONAL_HEADER, 'SizeOfImage', 0)) / 1e6
        ]
        pe.close()
        # 再次检查 meta 长度
        if len(meta) != 10:
            meta = (meta + [0]*10)[:10]
            
        return bytes_seq, meta

    except Exception:
        # 异常情况下返回固定维度的全零/填充数据
        return [256] * seq_len, [0.0] * 10

CWE_LIST = ['CWE-023', 'CWE-036', 'CWE-078', 'CWE-121', 'CWE-122', 'CWE-123', 'CWE-124', 'CWE-126', 'CWE-127', 'CWE-134', 'CWE-135', 'CWE-188', 'CWE-190', 'CWE-191', 'CWE-194', 'CWE-195', 'CWE-196', 'CWE-197', 'CWE-242', 'CWE-252', 'CWE-253', 'CWE-364', 'CWE-367', 'CWE-369', 'CWE-377', 'CWE-390', 'CWE-391', 'CWE-396', 'CWE-397', 'CWE-398', 'CWE-400', 'CWE-401', 'CWE-404', 'CWE-415', 'CWE-416', 'CWE-426', 'CWE-427', 'CWE-457', 'CWE-459', 'CWE-464', 'CWE-467', 'CWE-468', 'CWE-469', 'CWE-475', 'CWE-476', 'CWE-478', 'CWE-479', 'CWE-480', 'CWE-481', 'CWE-482', 'CWE-483', 'CWE-484', 'CWE-510', 'CWE-511', 'CWE-526', 'CWE-546', 'CWE-561', 'CWE-562', 'CWE-563', 'CWE-570', 'CWE-571', 'CWE-587', 'CWE-588', 'CWE-590', 'CWE-605', 'CWE-606', 'CWE-617', 'CWE-665', 'CWE-666', 'CWE-672', 'CWE-674', 'CWE-675', 'CWE-676', 'CWE-680', 'CWE-681', 'CWE-685', 'CWE-688', 'CWE-690', 'CWE-758', 'CWE-761', 'CWE-762', 'CWE-773', 'CWE-775', 'CWE-789', 'CWE-835', 'CWE-843']
CWE_TO_IDX = {cwe: i for i, cwe in enumerate(CWE_LIST)}
IDX_TO_CWE = {i: cwe for i, cwe in enumerate(CWE_LIST)}
