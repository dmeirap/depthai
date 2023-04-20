# Semantic Segmentation with deep

Requisitos:
tensorflow==1.15
tf_slim==1.0.0
Pillow

1- Etiquetar imagenes y crear una carpeta en /depthai/deeplab/datasets
    
   Ej: /adl_pcami_training_etquetas
        -/exp
          -/train_on_trainval_set
            -/eval
            -/export
            -/init_models
              -/deeplabv3_adl_pcami_train_aug #opcion de añadir un modelo previamente entrenado
            -/train
            -/vis
        -/ImageSets #Definir las imagenes para entrenar y evaluar
        -/JPEGImages  #Imagenes reales 240x135
        -/SegmentationClass #Imagenes segmentadas 240x135

2- Definir el dataset en data_generator.py
    
    _ADL_PCAMI_TRAINING_ETIQUETAS_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 5450, # number of file in the train folder
        'trainval': 4630 ,
        'val': 820,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
    )
    
    _DATASETS_INFORMATION = {
    'adl_pcami_training_etiquetas':_ADL_PCAMI_TRAINING_ETIQUETAS_INFORMATION,
    }

3- cd /depthai/deeplab/datasets
   python3 label_adl.py #Cambiar las rutas en funcion de la carpeta creada
   python3 build_adl_pcami_data.py #Cambiar las rutas en funcion de la carpeta creada 
   
4- Entrenar el modelo
    cd /depthai
    sh train_adl_pcami_training.sh #Cambiar las rutas y parametros en funcion de la carpeta creada
    sh eval_adl_pcami_training.sh  #Cambiar las rutas y parametros en funcion de la carpeta creada
      - Fijarse en el porcentaje de acierto del modelo 
        Ej: eval/miou_1.0_class_0[0.956388652]
            eval/miou_1.0_class_1[0.730497122]
            eval/miou_1.0_class_2[0.689852238]
	          eval/miou_1.0_overall[0.792246044]
    sh vis_adl_pcami_training.sh  #Cambiar las rutas y parametros en funcion de la carpeta creada
      - Resultados en /depthai/deeplab/datasets/adl_pcami_training_etiquetas/exp/train_on_trainval_set/vis/segmentation_results

5- Exportar modelo

    sh export_model.sh #Cambiar las rutas y parametros en funcion de la carpeta creada

6- Convertir el modelo en .bin y .xml
    
    ######### Solo la primera vez ########
    sudo apt-get install -y pciutils cpio
    sudo apt autoremove
    wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/v1.10.4/l_openvino_toolkit_p_2021.4.582.tgz
    path = "l_openvino_toolkit_p_2021.4.582.tgz"
    tar xf "{path}"
    cd l_openvino_toolkit_p_2021.4.582/
    ./install_openvino_dependencies.sh && \ sed -i 's/decline/accept/g' silent.cfg && \ ./install.sh --silent silent.cfg
    bash /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh
    ######################################
    
    source /opt/intel/openvino_2021/bin/setupvars.sh
    python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py \
    --input_model "/home/adl/adl/training/deeplab/datasets/adl_pcami_training_etiquetas/exp/train_on_trainval_set/export/frozen_adl_pcami_training_etiquetas.pb" \ #Cambiar las rutas y parametros en funcion de la carpeta creada
    --model_name adl_pcami_training_etiquetas \ #nombre que le damos al modelo
    --data_type FP16 \
    --input_shape [1,240,240,3] \
    --reverse_input_channel \
    --output_dir /home/adl/adl/training/models_optimizer #carpeta donde queremos que se guarde

7- Arreglar el archivo .xml para crear el .blob #Cambiar element_type = "i32" en lugar de "i64"
    
    Buscar el id="490" y copiar el nombre (ej:"strided_slice_10/extend_end_const1245431561")
    
      python3
    
      import xml.etree.ElementTree as ET
      tree = ET.parse("/home/adl/adl/training/models_optimizer/model_optimizer_adl_pcami_training_etiquetas/adl_pcami_training_etiquetas.xml") #Carpeta donde está nuestro .xml
      root = tree.getroot()
      data = root.find('.//layer[@name="strided_slice_10/extend_end_const1245431294"]/data') #Cambiar el nombre por el copiado
      data.set("element_type", "i32")
      tree.write("/home/adl/adl/training/models_optimizer/model_optimizer_adl_pcami_training_etiquetas/adl_pcami_training_etiquetas.xml")
     
8- Crear el archivo .blob para correr el modelo en la camara OAK
    
    Ir a http://blobconverter.luxonis.com/
    Seleccionar la versión 2021.4 y OpenVino Model
    Adjuntar los archivos .xml y .bin y definir shaves=6
    Convertir modelo y guardarlo en /depthai/oak/models.
    
9- Para correr el modelo en la cámara
    
    cd /depthai/oak
    python3 Segmentacion_pcami.py 
      -blob = dai.OpenVINO.Blob("/home/adl/adl/depthai/models/adl_pcami_training_etiquetas.blob")
 
