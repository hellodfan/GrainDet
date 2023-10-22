from xml.dom.minidom import parse
import xml.dom.minidom as xmldom
import copy
import os
import cv2
import numpy as np


grain_name2inx = {
            'wheat':{'NOR':0, 'FS':1, 'SD':2, 'MY':3, 'AP':4, 'BN':5, 'BP':6, 'IM':7},
            'sorg':{'NOR':0, 'FS':1, 'SD':2, 'MY':3, 'AP':4, 'BN':5, 'HD':6, 'IM':7},
            'rice':{'NOR':0, 'FS':1, 'SD':2, 'MY':3, 'AP':4, 'BN':5, 'UN':6, 'IM':7}
            }


def parse_xml(xml_path):

    gtype = xml_path.split('/')[-5]
    name2inx = grain_name2inx[gtype]

    # if 'F&S' in xml_path:
    #     with open(xml_path, 'r') as f:
    #         lines = f.readlines()
    #         new_lines = [l.replace('F&S','FS') if 'F&S' in l else l  for l in lines]
        
        # with open(xml_path, 'w') as f:
        #     f.writelines(new_lines)

    xmlfilepath = xml_path
    domobj = xmldom.parse(xmlfilepath)
    elementobj = domobj.documentElement

    # ============= GET CATEGORY============
    rec_result = []
    sub_element_obj = elementobj.getElementsByTagName('DU_grain')
    for i in range(len(sub_element_obj)):
        label = name2inx[sub_element_obj[i].firstChild.wholeText]
        rec_result.append(label)

    # ============= GET BBOX============
    box_result = []
    sub_element_obj = elementobj.getElementsByTagName('bndbox')
    for i in range(len(sub_element_obj)):
        bndbox = [0, 0, 0, 0]
        bndbox[0] = int(float(sub_element_obj[i].getElementsByTagName('x')[0].firstChild.data))
        bndbox[1] = int(float(sub_element_obj[i].getElementsByTagName('y')[0].firstChild.data))
        bndbox[2] = int(float(sub_element_obj[i].getElementsByTagName('w')[0].firstChild.data)) + bndbox[0]
        bndbox[3] = int(float(sub_element_obj[i].getElementsByTagName('h')[0].firstChild.data)) + bndbox[1]
        box_result.append(bndbox)

    # ============= GET MASK============
    sub_element_obj = elementobj.getElementsByTagName('mask')
    mask_result = []
    for i in range(len(sub_element_obj)):
        contors = []
        for j in range(len(sub_element_obj[i].getElementsByTagName('x'))):
            x = int(float(sub_element_obj[i].getElementsByTagName('x')[j].firstChild.data))
            y = int(float(sub_element_obj[i].getElementsByTagName('y')[j].firstChild.data))
            contors.append([x,y])
        
        contors = np.array(contors)
        msk = np.zeros((3644, 5480), dtype=np.uint8)
        msk = cv2.fillPoly(msk, [contors], 1)
        mask_result.append(msk)

    return box_result, mask_result, rec_result


def merge_box(ubox, dbox, rec_result):
    res = []
    for ub, db, label in zip(ubox, dbox, rec_result):
        ux1, uy1, ux2, uy2 = ub
        dx1, dy1, dx2, dy2 = db
        mbox = [label, min(ux1, dx1), min(uy1, dy1), max(ux2, dx2), max(uy2, dy2)]
        res.append(mbox)
    return res


def merge_mask(umask, dmask, rec_result):
    res = []
    for um, dm, label in zip(umask, dmask, rec_result):
        msk = np.zeros((3644, 5480), dtype=np.uint8)
        union = um + dm
        msk[union>0] = label+1
        res.append(msk)
    return res


def get_txt_npz(xml_path, save_path, gtype):

    phase = xml_path.split('/')[-3]

    save_txt_dir = os.path.join(save_path, gtype, 'txt', phase)
    save_npz_dir = os.path.join(save_path, gtype, 'npz', phase)

    os.makedirs(save_txt_dir, exist_ok=True)
    os.makedirs(save_npz_dir, exist_ok=True)

    fname = xml_path.split('/')[-1].replace('_U_p600s.xml','_UD_p600s')
    txt_path = os.path.join(save_txt_dir, fname+'.txt')
    npz_path = os.path.join(save_npz_dir, fname)

    if not os.path.exists(npz_path+'.npz'):

        ubox_result, umask_result, urec_result = parse_xml(xml_path)
        dbox_result, dmask_result, drec_result = parse_xml(xml_path.replace('_U_','_D_'))

        m_boxes = merge_box(ubox_result, dbox_result, urec_result)
        m_masks = merge_mask(umask_result, dmask_result, urec_result)

        f = open(txt_path,'w+')
        
        for box in m_boxes:
            f.write(' '.join(list(map(str, box))) + '\n')
        f.close()

        np.savez_compressed(npz_path, np.stack(m_masks, axis=0))  # np.load('*.npz')['arr_0']
    
    else:

        print(npz_path+'.npz is exist! Process next...')


if __name__=='__main__':
    xmls_dir = '/opt/data1/docker/data/test_1016/rice'
    save_dir =  '/opt/data1/docker/data/test_1016/'
    for root, dir, files in os.walk(xmls_dir):
        for ix, f in enumerate(files):
            if '_U_p600s.xml' in f:
                f_path = os.path.join(root, f)
                print(f'Processing {f_path}....{ix}/{len(files)}')
                get_txt_npz(f_path, save_dir, xmls_dir.split('/')[-1])
