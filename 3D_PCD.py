import numpy as np
from get_Matrix import get_ExtrinsicMatrix, get_IntrinsicMatrix
from get_points import main
import open3d as o3d

def get_3D_PCD(sequence, lines):
    img_3Dpts = []
    for line in lines:
        ts = line.strip("\n")
        dataset_path = f'{sequence}/dataset/{ts}'
        cam_info_path = './camera_info/lucid_cameras_x00'
        ExtMatrix = get_ExtrinsicMatrix(cam_info_path)
        IntMatrix = get_IntrinsicMatrix(cam_info_path)

        cam_direction = np.loadtxt(f'{dataset_path}/camera.csv', dtype=str, delimiter=',')
        cam_direction = str(cam_direction).split('/')[2].split('_')[2]

        img_2Dpts = main(dataset_path, ts)
        
        for img_2Dpt in img_2Dpts:
            img_2Dpt = np.array([img_2Dpt[0], img_2Dpt[1], 1]).T  # [x, y, 1]
            
            if cam_direction == 'f':
                R_bl_f, t_bl_f = ExtMatrix['base_link-f'][:, :3], ExtMatrix['base_link-f'][:, -1:]
                t_bl_f = np.reshape(t_bl_f, -1).T

                camBased_matrix = np.dot(np.linalg.inv(IntMatrix['f']), img_2Dpt)
                img_3Dpt = np.dot(R_bl_f, camBased_matrix) + t_bl_f
                img_3Dpts.append(img_3Dpt / np.linalg.norm(img_3Dpt))

            if cam_direction == 'fl':
                R_bl_f, t_bl_f = ExtMatrix['base_link-f'][:, :3], np.reshape(ExtMatrix['base_link-f'][:, -1:], -1).T
                R_f_fl, t_f_fl = ExtMatrix['f-fl'][:, :3], np.reshape(ExtMatrix['f-fl'][:, -1:], -1).T

                camBased_matrix = np.dot(np.linalg.inv(IntMatrix['fl']), img_2Dpt)
                tmp_3Dpt = np.dot(R_f_fl, camBased_matrix) + t_f_fl
                img_3Dpt = np.dot(R_bl_f, tmp_3Dpt) + t_bl_f
                img_3Dpts.append(img_3Dpt / np.linalg.norm(img_3Dpt))

            if cam_direction == 'fr':
                R_bl_f, t_bl_f = ExtMatrix['base_link-f'][:, :3], np.reshape(ExtMatrix['base_link-f'][:, -1:], -1).T
                R_f_fr, t_f_fr = ExtMatrix['f-fr'][:, :3], np.reshape(ExtMatrix['f-fr'][:, -1:], -1).T

                camBased_matrix = np.dot(np.linalg.inv(IntMatrix['fr']), img_2Dpt)
                tmp_3Dpt = np.dot(R_f_fr, camBased_matrix) + t_f_fr
                img_3Dpt = np.dot(R_bl_f, tmp_3Dpt) + t_bl_f
                img_3Dpts.append(img_3Dpt / np.linalg.norm(img_3Dpt))

            if cam_direction == 'b':
                R_bl_f, t_bl_f = ExtMatrix['base_link-f'][:, :3], np.reshape(ExtMatrix['base_link-f'][:, -1:], -1).T
                R_f_fl, t_f_fl = ExtMatrix['f-fl'][:, :3], np.reshape(ExtMatrix['f-fl'][:, -1:], -1).T
                R_fl_b, t_fl_b = ExtMatrix['fl-b'][:, :3], np.reshape(ExtMatrix['fl-b'][:, -1:], -1).T

                camBased_matrix = np.dot(np.linalg.inv(IntMatrix['b']), img_2Dpt)
                tmp_3Dpt = np.dot(R_fl_b, camBased_matrix) + t_fl_b
                tmp_3Dpt = np.dot(R_f_fl, tmp_3Dpt) + t_f_fl
                img_3Dpt = np.dot(R_bl_f, tmp_3Dpt) + t_bl_f
                img_3Dpts.append(img_3Dpt / np.linalg.norm(img_3Dpt))


    np.savetxt('./img_3Dpts.txt', img_3Dpts)
    pcd = o3d.io.read_point_cloud('./img_3Dpts.txt', format='xyz')
    # print(pcd)
    o3d.visualization.draw_geometries([pcd])

    return pcd

if __name__ =='__main__':
    sequence = 'seq1'
    txt_path = "seq1/all_timestamp.txt"
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    img_3d_pcd = get_3D_PCD(sequence, lines)