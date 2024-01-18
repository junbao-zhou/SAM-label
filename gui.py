import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import typing
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QMainWindow, QTextEdit, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QFileSystemModel, QListView, QSplitter, QLineEdit, QTableView

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import matplotlib as mpl
from PIL import Image
import xml.etree.ElementTree as ET
import pathlib
from glob import glob
import imghdr


images_path = "<your-path-to-input-images>"

sam_checkpoint = "pre-training/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](
    checkpoint=sam_checkpoint).to(device=device)

predictor = SamPredictor(sam)


def get_3_bits(num):
    return num & 0b111


def get_bit(num, pos):
    return (num & (1 << pos)) >> pos


def get_color(index):
    first_3_bits = get_3_bits(index)
    second_3_bits = get_3_bits(index >> 3)
    R = (get_bit(first_3_bits, 0) << 1) + get_bit(second_3_bits, 0)
    G = (get_bit(first_3_bits, 1) << 1) + get_bit(second_3_bits, 1)
    B = (get_bit(first_3_bits, 2) << 1) + get_bit(second_3_bits, 2)
    R = R << 6
    G = G << 6
    B = B << 6
    return [R, G, B]


class HideFileTypesProxy(QSortFilterProxyModel):
    """
    A proxy model that excludes files from the view
    that end with the given extension
    """

    def __init__(self, excludes, *args, **kwargs):
        super(HideFileTypesProxy, self).__init__(*args, **kwargs)
        self._excludes = excludes[:]

    def filterAcceptsRow(self, srcRow, srcParent):
        idx = self.sourceModel().index(srcRow, 0, srcParent)
        name = idx.data()

        # Can do whatever kind of tests you want here,
        # against the name
        for exc in self._excludes:
            if name.endswith(exc):
                return False

        return True


class FileDataModel(QFileSystemModel):
    def data(self, index: QModelIndex, role: int = ...):
        # print(f"{index.row() = }  {index.column() = }  {role = }")
        index_0 = index.siblingAtColumn(0)
        data = super().data(index_0, role)
        # print(f"{data = }")
        if index.column() == 0:
            return data
        if data is not None and isinstance(data, str):
            if index.column() == 1:
                label_png = glob(os.path.join(self.rootPath(), f"{data}.*.label.png"))
                # print(f"{label_png = }")
                if len(label_png) > 0:
                    return pathlib.Path(label_png[0]).name
                else:
                    return None
            elif index.column() == 2:
                label_xml = glob(os.path.join(self.rootPath(), f"{data}.*.xml"))
                # print(f"{label_xml = }")
                if len(label_xml) > 0:
                    return pathlib.Path(label_xml[0]).name
                else:
                    return None
            else:
                return None
        return None

    def columnCount(self, parent: QModelIndex = ...) -> int:
        columnCount = super().columnCount(parent)
        return 3

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> typing.Any:
        header = super().headerData(section, orientation, role)
        return header


class ClickLabel(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_mask = None
        self.points = []
        self.input_points = []
        self.input_labels = []
        self.masks = []
        self._palette = [c for i in range(32) for c in get_color(i)]
        self.file_name = None
        self.predicted_logit = None

    def mousePressEvent(self, ev) -> None:
        x = ev.pos().x()
        y = ev.pos().y()
        # print(f"click on position : {x}, {y}")
        self.points.append(ev.pos())
        self.input_points.append([x, y])
        self.input_labels.append(1 if ev.button() == Qt.LeftButton else 0)
        mask, score, logit = predictor.predict(
            point_coords=np.array(self.input_points),
            point_labels=np.array(self.input_labels),
            mask_input=self.predicted_logit,
            multimask_output=False,
        )
        self.predicted_logit = logit
        self.current_mask = mask.squeeze()
        self.update()
        self.parent().set_mask(self.masks + [self.current_mask])

    def paintEvent(self, ev) -> None:
        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(8)
        for p, label in zip(self.points, self.input_labels):
            pen.setColor(Qt.GlobalColor.darkYellow if label == 1 else Qt.GlobalColor.darkBlue)
            painter.setPen(pen)
            painter.drawPoint(p)

    def clear_current_input(self):
        self.current_mask = None
        self.points.clear()
        self.input_points.clear()
        self.input_labels.clear()
        self.predicted_logit = None
        self.update()
        self.parent().set_mask(self.masks)

    def reset(self, w, h, image_file: str):
        self.masks.clear()
        self.resize(w, h)
        self.clear_current_input()
        self.file_name = image_file

    def complete_one_mask(self):
        if self.current_mask is not None:
            self.masks.append(self.current_mask.copy())
        self.clear_current_input()

    def save_masks(self, label_name: str):
        assert self.file_name is not None
        if len(self.masks) == 0:
            return
        mask = np.zeros(self.masks[0].shape, dtype=np.uint8)
        for i, m in enumerate(self.masks, start=1):
            mask = np.where(m, i, mask)
        save_image = Image.fromarray(mask).convert('P')
        save_image.putpalette(self._palette)
        save_file_name = f"{self.file_name}.{label_name}.label.png"
        print(f"Saving mask {save_file_name}")
        save_image.save(save_file_name)

        data = ET.Element('annotation')
        folder = ET.SubElement(data, 'folder')
        folder.text = str(pathlib.Path(self.file_name).parents[0])
        filename = ET.SubElement(data, 'filename')
        filename.text = str(pathlib.Path(self.file_name).name)
        path = ET.SubElement(data, 'path')
        path.text = self.file_name
        size = ET.SubElement(data, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        width.text = f"{self.width()}"
        height.text = f"{self.height()}"
        depth.text = f"{3}"
        segmented = ET.SubElement(data, 'segmented')
        segmented.text = f"{0}"
        for m in self.masks:
            obj = ET.SubElement(data, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = label_name
            pose = ET.SubElement(obj, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = f"{0}"
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = f"{0}"
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')
            x, y = np.where(m)
            xmin.text = f"{x.min()}"
            xmax.text = f"{x.max()}"
            ymin.text = f"{y.min()}"
            ymax.text = f"{y.max()}"
        ET.indent(data, space="\t", level=0)
        with open(f"{self.file_name}.{label_name}.xml", "wb") as f:
            f.write(ET.tostring(data))
        # data.write(f"{self.file_name}.{label_name}.xml", encoding="utf-8")


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.mask_label = QLabel(self)

        self.click_label = ClickLabel(self)

    def set_image(self, image_file: str):
        self.clear()

        cv_image = cv2.imread(image_file)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        print(f"cv2 read image {image_file} \n {cv_image.shape[0]}x{cv_image.shape[1]}")

        predictor.set_image(cv_image)
        print(f"Predictor set image completed")

        self.image = QPixmap(image_file)
        self.setPixmap(self.image)
        self.resize(self.image.width(), self.image.height())
        print(f"Set Image w {self.image.width()}  h {self.image.height()}")

        self.mask_label.resize(self.image.width(), self.image.height())
        # self.set_mask([np.zeros((self.image.height(), self.image.width()), dtype=bool)])
        self.set_mask([])

        self.click_label.reset(self.image.width(), self.image.height(), image_file=image_file)

    def set_mask(self, np_masks: np.ndarray):
        mask = np.zeros((self.image.height(), self.image.width(), 4), dtype=np.uint8)
        np_mask = np.logical_or.reduce(np_masks, axis=0)
        np_mask = np.logical_not(np_mask).astype(np.uint8) * 90
        mask[..., 3] = np_mask
        for i, m in enumerate(np_masks):
            contours, hierarchy = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                cv2.drawContours(mask, contours, -1, get_color(i + 1) + [200], 3)
        self.mask_label.setPixmap(
            QPixmap(QImage(mask, self.image.width(), self.image.height(), QImage.Format_RGBA8888)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        centeral_widget = QWidget(self)
        layout = QHBoxLayout()
        centeral_widget.setLayout(layout)
        self.setCentralWidget(centeral_widget)

        h_splitter = QSplitter(Qt.Horizontal)

        image_layout = QVBoxLayout()

        self.label_text = QLineEdit()
        image_layout.addWidget(self.label_text)

        self.img_label = ImageLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.img_label)

        image_layout.addWidget(scroll_area)

        buttons_layout = QHBoxLayout()

        save_object_button = QPushButton("save object")
        save_object_button.clicked.connect(self.img_label.click_label.complete_one_mask)
        buttons_layout.addWidget(save_object_button)

        save_mask_button = QPushButton("save masks")
        save_mask_button.clicked.connect(
            lambda: self.img_label.click_label.save_masks(self.label_text.text()))
        buttons_layout.addWidget(save_mask_button)

        clear_current_button = QPushButton("clear current")
        clear_current_button.clicked.connect(self.img_label.click_label.clear_current_input)
        buttons_layout.addWidget(clear_current_button)

        image_layout.addLayout(buttons_layout)

        image_widget = QWidget()
        image_widget.setLayout(image_layout)
        h_splitter.addWidget(image_widget)

        self.file_model = FileDataModel()
        self.file_model.setRootPath(images_path)
        self.proxyModel = HideFileTypesProxy(excludes=[".label.png", ".xml", ".npy"], parent=self)
        self.proxyModel.setDynamicSortFilter(True)
        self.proxyModel.setSourceModel(self.file_model)
        self.file_list_view = QTableView()
        self.file_list_view.setModel(self.proxyModel)
        self.file_list_view.setRootIndex(
            self.proxyModel.mapFromSource(self.file_model.index(images_path)))
        self.file_list_view.selectionModel().selectionChanged.connect(
            self.file_selected
        )

        h_splitter.addWidget(self.file_list_view)

        layout.addWidget(h_splitter)

    def file_selected(self):
        for ix in self.file_list_view.selectedIndexes():
            print(f"Table select {ix.row() = }  {ix.column() = }")
            value = ix.sibling(ix.row(), ix.column()).data()
            print(f"Table select {value = }")
            if not isinstance(value, str):
                return
            file_path = os.path.join(self.file_model.rootPath(), value)
            if imghdr.what(file_path) is None:
                return
            self.img_label.set_image(file_path)
            return


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
