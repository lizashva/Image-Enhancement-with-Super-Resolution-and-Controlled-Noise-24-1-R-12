import React, { useState } from "react";
import axios from "axios";

const RestoreImage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [lowResUrl, setLowResUrl] = useState<string | null>(null);
  const [highResUrl, setHighResUrl] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      const fileURL = URL.createObjectURL(event.target.files[0]);
      setLowResUrl(fileURL); // Preview the uploaded (low-res) image immediately
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:5000/restore-image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Server responded with:", response.data);

      setHighResUrl(
        `http://localhost:5000/${response.data.high_resolution_image_path}`
      );
      setLowResUrl(
        `http://localhost:5000/${response.data.low_resolution_image_path}`
      );
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed! Check the console for more details.");
    }
  };

  return (
    <div className="flex flex-col items-center p-4 space-y-3">
      <input
        type="file"
        className="file-input mb-4"
        onChange={handleFileChange}
      />
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        onClick={handleUpload}
      >
        Start process
      </button>
      {lowResUrl && (
        <div className="mt-4">
          <h3>Low Resolution Image</h3>
          <img src={lowResUrl} alt="Low Resolution" />
        </div>
      )}
      {highResUrl && (
        <div className="mt-4">
          <h3>High Resolution Image</h3>
          <img src={highResUrl} alt="High Resolution" />
        </div>
      )}
    </div>
  );
};

export default RestoreImage;
