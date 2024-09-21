import React, { useState } from 'react';
import axios from 'axios';
import Breadcrumb from './NewDesign/Breadcrumbs/Breadcrumb';

const TrainModels: React.FC = () => {
  const [model, setModel] = useState('Noise2Noise');
  const [dataset, setDataset] = useState('dataset/data_set1');
  const [noiseType, setNoiseType] = useState('Poisson');
  const [trainSize, setTrainSize] = useState('');
  const [validSize, setValidSize] = useState('');
  const [nbEpochs, setNbEpochs] = useState('');
  const [loss, setLoss] = useState('L1');
  const [noiseParam, setNoiseParam] = useState('0');
  const [cropSize, setCropSize] = useState('128');
  const [status, setStatus] = useState({ message: '', type: '' });

  const validateTrainingData = () => {
    setStatus({ message: '', type: '' });
    console.log(
      model,
      dataset,
      noiseType,
      trainSize,
      validSize,
      nbEpochs,
      loss,
      noiseParam,
      cropSize,
    );
    if (
      !model ||
      !dataset ||
      !noiseType ||
      !trainSize ||
      !validSize ||
      !nbEpochs ||
      !loss ||
      !noiseParam ||
      !cropSize
    ) {
      setStatus({ message: 'All fields must be filled out.', type: 'error' });
      return false;
    }

    if (isNaN(parseInt(trainSize)) || parseInt(trainSize) <= 0) {
      setStatus({
        message: 'Training size must be a positive number.',
        type: 'error',
      });
      return false;
    }

    if (isNaN(parseInt(validSize)) || parseInt(validSize) <= 0) {
      setStatus({
        message: 'Validation size must be a positive number.',
        type: 'error',
      });
      return false;
    }

    if (isNaN(parseInt(nbEpochs)) || parseInt(nbEpochs) <= 0) {
      setStatus({
        message: 'Number of epochs must be a positive number.',
        type: 'error',
      });
      return false;
    }

    return true;
  };

  const handleTrain = async () => {
    if (!validateTrainingData()) {
      console.log(status.message);
      return;
    }
    try {
      let trainingData: any = {
        model,
        dataset,
        noiseType,
        trainSize,
        validSize,
        nbEpochs,
        loss,
        noiseParam,
        cropSize,
      };

      const url = 'http://localhost:5000/train-model';

      if (model === 'Super resolution') {
        // מחיקת המאפיינים noiseType ו-noiseParam עבור דגם Super resolution

        trainingData = {
          model,
          dataset,
          trainSize,
          validSize,
          nbEpochs,
          loss,
          cropSize,
        };
      }
      const response = await axios.post(url, trainingData, {
        headers: { 'Content-Type': 'application/json' },
      });
      console.log('Server responded with:', response.data);
      setStatus({ message: 'Training request sent.', type: 'success' });

      // Reset form after success
      setModel('Noise2Noise');
      setDataset('dataset/data_set1');
      setNoiseType('Poisson');
      setTrainSize('');
      setValidSize('');
      setNbEpochs('');
      setLoss('L1');
      setNoiseParam('0');
      setCropSize('128');
    } catch (error) {
      console.error('Training request error:', error);
      setStatus({ message: 'Training request failed!', type: 'error' });
    }
  };

  return (
    <div className="col-span-5 xl:col-span-3">
      <Breadcrumb pageName="Train Models" />
      <div className="rounded-sm border border-stroke bg-white shadow-default dark:border-strokedark dark:bg-boxdark">
        <div className="border-b border-stroke py-4 px-7 dark:border-strokedark">
          <h3 className="font-medium text-black dark:text-white">
            Train Models
          </h3>
        </div>
        <div className="p-7">
          <form onSubmit={(e) => e.preventDefault()}>
            {/* Model Selection */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Type of Model
              </label>
              <select
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                <option value="Super resolution">Super resolution</option>
                <option value="Noise2Noise">Noise2Noise</option>
              </select>
            </div>

            {/* Dataset Selection */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Dataset (train + test)
              </label>
              <select
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
              >
                <option value="dataset/data_set1">dataset/data_set1</option>
                <option value="dataset/data_set2">dataset/data_set2</option>
                <option value="dataset/data_set3">dataset/data_set3</option>
              </select>
            </div>

            {/* Noise Type */}
            {model === 'Noise2Noise' && (
              <div className="mb-5.5">
                <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                  Type of Noise
                </label>
                <select
                  className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                  value={noiseType}
                  onChange={(e) => setNoiseType(e.target.value)}
                >
                  <option value="Gaussian">Gaussian</option>
                  <option value="Poisson">Poisson</option>
                  <option value="Multiplicative Bernoulli">
                    Multiplicative Bernoulli
                  </option>
                  <option value="Random text">Random text</option>
                </select>
              </div>
            )}

            {/* Training Size */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Training Size
              </label>
              <input
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                type="text"
                value={trainSize}
                onChange={(e) => setTrainSize(e.target.value)}
              />
            </div>

            {/* Validation Size */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Validation Size
              </label>
              <input
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                type="text"
                value={validSize}
                onChange={(e) => setValidSize(e.target.value)}
              />
            </div>

            {/* Epochs */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Number of Epochs
              </label>
              <input
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                type="text"
                value={nbEpochs}
                onChange={(e) => setNbEpochs(e.target.value)}
              />
            </div>

            {/* Loss */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Loss
              </label>
              <select
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                value={loss}
                onChange={(e) => setLoss(e.target.value)}
              >
                <option value="L1">L1</option>
                <option value="L2">L2</option>
                <option value="Lhdr">Lhdr</option>
              </select>
            </div>

            {/* Noise Parameter */}
            {model === 'Noise2Noise' && (
              <div className="mb-5.5">
                <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                  Noise Parameter
                </label>
                <input
                  className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                  type="text"
                  value={noiseParam}
                  onChange={(e) => setNoiseParam(e.target.value)}
                />
              </div>
            )}

            {/* Crop Size */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Crop Size
              </label>
              <select
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                value={cropSize}
                onChange={(e) => setCropSize(e.target.value)}
              >
                <option value="128">128</option>
                <option value="256">256</option>
              </select>
            </div>

            <div className="flex justify-end gap-4.5">
              <button
                className="flex justify-center rounded border border-stroke py-2 px-6 font-medium text-black hover:shadow-1 dark:border-strokedark dark:text-white"
                type="button"
                onClick={() => {
                  setModel('Noise2Noise');
                  setDataset('dataset/data_set1');
                  setNoiseType('Poisson');
                  setTrainSize('');
                  setValidSize('');
                  setNbEpochs('');
                  setLoss('L1');
                  setNoiseParam('');
                  setCropSize('');
                }}
              >
                Reset
              </button>
              <button
                className="flex justify-center rounded bg-primary py-2 px-6 font-medium text-white hover:bg-opacity-90"
                type="button"
                onClick={handleTrain}
              >
                Train
              </button>
            </div>
          </form>
          {status.message && (
            <div
              className={` my-3 text-white px-4 py-3 rounded ${status.type === 'success' ? 'bg-green-500' : 'bg-red-500'}`}
            >
              {status.message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainModels;
