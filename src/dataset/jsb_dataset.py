import pypianoroll as pr
import numpy as np

from functools import cached_property
from dataclasses import dataclass
from copy import copy
import os.path
import json

from typing import List

from .dataset import Dataset

path = os.path.join(os.path.dirname(__file__), "jsb-chorales-16th.json")

@dataclass
class JSBDatasetInfo:
    min_pitch: int = 36
    max_pitch: int = 81
    resolution: float = 16 # 16th notes
    piece_length: int = 64
    bpm: int = 15

    @property
    def num_pitches(self):
        return self.max_pitch - self.min_pitch + 1

    def save_pianoroll(self, pianoroll: np.ndarray, filename: str):
        """save pianoroll to midi file

        Args:
            pianoroll (np.ndarray): numpy array of shape (num_timesteps, num_pitches, num_tracks)
            filename (str): midi filename
        """
        pianoroll = np.pad(pianoroll, ((0, 0), (self.min_pitch, 127-self.max_pitch)), mode="constant", constant_values=0)
        track = pr.BinaryTrack(pianoroll=pianoroll)
        multitrack = pr.Multitrack(tracks=[track], tempo=self.bpm, resolution=self.resolution // 2)
        multitrack.write(filename)
    
    def to_pianoroll(
        self,
        piece: List[List[int]],
    ) -> np.ndarray:
        """convert piece to binary pianoroll

        Args:
            piece (List[List[int]]): each element of the list is a chord, each chord is a list of pitches so the list should be of shape (num_timesteps, num_tracks)

        Returns:
            np.ndarray: binary pianoroll of shape (num_timesteps, num_pitches, num_tracks)
        """        
        pianoroll = np.zeros((len(piece), self.num_pitches), dtype=bool)
        for i, chord in enumerate(piece):
            for track in chord:
                assert self.min_pitch <= track <= self.max_pitch, "pitch out of range"
                pianoroll[i, track - self.min_pitch] = True
        return pianoroll


class Jsb16thSeparatedDataset(Dataset):
    def __init__(self, data: List[List[List[int]]], info: JSBDatasetInfo = JSBDatasetInfo()):
        self.info = copy(info)

        # save all the info as attributes
        self.min_pitch = info.min_pitch
        self.max_pitch = info.max_pitch
        self.resolution = info.resolution
        self.qpm = info.bpm

        # convert each piece to a pianoroll
        super().__init__([self.info.to_pianoroll(piece) for piece in data])

    def _random_crop(self, pianoroll: np.ndarray) -> np.ndarray:
        if len(pianoroll) < self.info.piece_length:
            raise ValueError(f"Piece length is too short: {len(pianoroll)}")
        # pick a random start index (the piece starts, rather than ends with an upbeat)
        start = np.random.choice(
            np.arange(
                len(pianoroll) % self.info.piece_length,
                len(pianoroll),
                self.info.piece_length,
            )
        )
        return pianoroll[start : start + self.info.piece_length]

    def augment(self, data: np.ndarray) -> np.ndarray:
        return self._random_crop(data)

class Jsb16thSeparatedDatasetFactory:
    """create datasets from a json file"""
    def __init__(self, path: str = path, info: JSBDatasetInfo = JSBDatasetInfo()):
        with open(path) as f:
            self.data = json.load(f)
        self.info = info

    @cached_property
    def train_dataset(self) -> Jsb16thSeparatedDataset:
        return Jsb16thSeparatedDataset(self.data["train"], self.info)

    @cached_property
    def val_dataset(self) -> Jsb16thSeparatedDataset:
        return Jsb16thSeparatedDataset(self.data["valid"], self.info)

    @cached_property
    def test_dataset(self) -> Jsb16thSeparatedDataset:
        return Jsb16thSeparatedDataset(self.data["test"], self.info)