import React, { useState } from 'react';
import axios from 'axios';
import FullScreenSpinner from './fullScreenSpinner'; // Import the full-screen spinner
import Breadcrumb from './NewDesign/Breadcrumbs/Breadcrumb';

const RestoreImage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [lowResUrl, setLowResUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [operation, setOperation] = useState<string>('high-res'); // "high-res" or "remove-noise"
  const [loading, setLoading] = useState<boolean>(false); // Loading state
  const [noiseType, setNoiseType] = useState<string>(''); // Noise type state

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      const fileURL = URL.createObjectURL(event.target.files[0]);
      setLowResUrl(fileURL); // Preview the uploaded image immediately
      setResultUrl(null); // Clear the result image
    }
  };

  const handleOperationChange = (
    event: React.ChangeEvent<HTMLSelectElement>,
  ) => {
    setOperation(event.target.value);
    setFile(null); // Clear the file
    setLowResUrl(null); // Clear the uploaded image
    setResultUrl(null); // Clear the result image
    if (event.target.value === 'remove-noise') {
      setNoiseType(''); // Reset noise type when operation changes
    }
  };

  const handleNoiseTypeChange = (
    event: React.ChangeEvent<HTMLSelectElement>,
  ) => {
    setNoiseType(event.target.value);
    setFile(null); // Clear the file
    setLowResUrl(null); // Clear the uploaded image
    setResultUrl(null); // Clear the result image
  };

  const handleFileSelect = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/jpeg, image/png';
    fileInput.onchange = (event: any) => {
      const file = event.target.files[0];
      if (file) {
        setFile(file);
        const fileURL = URL.createObjectURL(file);
        setLowResUrl(fileURL); // Preview the uploaded image immediately
        setResultUrl(null); // Clear the result image
      }
    };
    fileInput.click(); // Open the file picker dialog
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select an image before starting the process.');
      return;
    }

    setLoading(true); // Set loading state to true

    const formData = new FormData();
    formData.append('file', file);
    formData.append('operation', operation);
    if (operation === 'remove-noise') {
      formData.append('noise_type', noiseType); // Include noise type if operation is "remove-noise"
    }

    const endpoint =
      operation === 'high-res' ? 'restore-image' : 'remove-noise';
    const url = `http://localhost:5000/${endpoint}`;

    try {
      const response = await axios.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Server responded with:', response.data);

      const resultImagePath =
        operation === 'high-res'
          ? response.data.high_resolution_image_path
          : response.data.noise_removed_image_path;
      setResultUrl(`http://localhost:5000/${resultImagePath}`);
    } catch (error) {
      console.error('Upload error:', error);
      alert('Upload failed! Check the console for more details.');
    } finally {
      setLoading(false); // Set loading state to false
    }
  };

  return (
    <div className="col-span-5 xl:col-span-3">
      <Breadcrumb pageName={`Restore Image`} />
      <div className="rounded-sm border border-stroke bg-white shadow-default dark:border-strokedark dark:bg-boxdark">
        <div className="border-b border-stroke py-4 px-7 dark:border-strokedark">
          <h3 className="font-medium text-black dark:text-white">
            Restore Image
          </h3>
        </div>
        <div className="p-7">
          <FullScreenSpinner loading={loading} />{' '}
          {/* Add the full-screen spinner */}
          <form onSubmit={(e) => e.preventDefault()}>
            {/* Operation Selection */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Choose Operation:
              </label>
              <select
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                value={operation}
                onChange={handleOperationChange}
              >
                <option value="high-res">Super Resolution</option>
                <option value="remove-noise">Remove Noise</option>
              </select>
            </div>

            {/* Noise Type Selection */}
            {operation === 'remove-noise' && (
              <div className="mb-5.5">
                <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                  Choose Noise Type:
                </label>
                <select
                  className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                  value={noiseType}
                  onChange={handleNoiseTypeChange}
                >
                  <option value="">Select Noise Type</option>
                  <option value="Gaussian">Gaussian</option>
                  <option value="Poisson">Poisson</option>
                  <option value="Multiplicative Bernoulli">
                    Multiplicative Bernoulli
                  </option>
                  <option value="Random Text">Random Text</option>
                </select>
              </div>
            )}

            {/* File Upload */}
            <div className="mb-5.5">
              <label className="mb-3 block text-sm font-medium text-black dark:text-white">
                Upload Image:
              </label>
              <button
                className="w-full rounded border border-stroke bg-gray py-3 px-4.5 text-black focus:border-primary focus-visible:outline-none dark:border-strokedark dark:bg-meta-4 dark:text-white dark:focus:border-primary"
                type="button"
                onClick={handleFileSelect}
              >
                Choose File
              </button>
              {file && (
                <p className="mt-2 text-sm text-black dark:text-white">
                  {file.name}
                </p>
              )}
            </div>

            <div className="flex justify-end gap-4.5">
              <button
                className="flex justify-center rounded bg-primary py-2 px-6 font-medium text-white hover:bg-opacity-90"
                type="button"
                onClick={handleUpload}
              >
                Start Process
              </button>
            </div>
          </form>
          {/* Display Images */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {lowResUrl && (
              <div>
                <h3 className="text-black font-semibold text-xl dark:text-white text-left mb-2">
                  Uploaded Image
                </h3>
                <img
                  src={lowResUrl}
                  alt="Uploaded"
                  className="rounded object-cover"
                  style={{ width: '400px', height: '400px' }}
                />
              </div>
            )}

            {resultUrl && (
              <div>
                <h3 className="text-black text-xl font-semibold dark:text-white text-left mb-2">
                  Result Image
                </h3>
                <img
                  src={resultUrl}
                  alt="Result"
                  className="rounded object-cover"
                  style={{ width: '400px', height: '400px' }}
                />
              </div>
            )}
          </div>
          {/* Download button */}
          {resultUrl && (
            <div className="my-3 ml-52 w-full flex justify-center">
              <a
                href={resultUrl}
                download="restored-image.jpg" // Name for the downloaded file
                className="rounded bg-blue-600 py-2 px-6 font-medium text-white hover:bg-blue-500"
              >
                Download Restored Image
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RestoreImage;
