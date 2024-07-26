import React, { useState } from "react";

interface TrainModelsProps {
  onTrain: (model: string, dataset: string, noiseType?: string) => void;
}

const TrainModels: React.FC = () => {
  const [model, setModel] = useState("Noise2Noise");
  const [dataset, setDataset] = useState("dataset/data_set1");
  const [noiseType, setNoiseType] = useState("Poisson");
  const onTrain = (model: any, dataset: any, noiseType: any) => {};
  return (
    <div className="p-4 space-y-3">
      <div>
        <label className="block text-gray-700 text-sm font-bold mb-2">
          Type of model:
        </label>
        <select
          className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="Super resolution">Super resolution</option>
          <option value="Noise2Noise">Noise2Noise</option>
        </select>
      </div>
      {model != "Super resolution" && (
        <div>
          <label className="block text-gray-700 text-sm font-bold mb-2">
            Type of noise (only for Noise2Noise):
          </label>
          <select
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            value={noiseType}
            onChange={(e) => setNoiseType(e.target.value)}
          >
            <option value="Gaussian">Gaussian</option>
            <option value="Poisson">Poisson</option>
            <option value="Multiplicative Bernoulli">
              Multiplicative Bernoulli
            </option>
            <option value="Random text">Random text</option>
            <option value="Random text">Random valued</option>
          </select>
        </div>
      )}
      <div>
        <label className="block text-gray-700 text-sm font-bold mb-2">
          Dataset (train + test):
        </label>
        <select
          className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
        >
          <option value="dataset/data_set1">dataset/data_set1</option>
        </select>
      </div>
      <button
        className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
        onClick={() => onTrain(model, dataset, noiseType)}
      >
        Train
      </button>
    </div>
  );
};

export default TrainModels;
