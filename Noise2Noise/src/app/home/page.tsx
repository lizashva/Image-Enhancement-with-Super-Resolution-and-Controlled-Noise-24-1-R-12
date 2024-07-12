"use client";
import axios from "axios";
import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files?.length) {
      setFile(event.target.files[0]);
    }
  };

  const onRestoreImage = async () => {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);

      try {
        // Adjust the URL as necessary to point to your Flask API
        const response = await axios.get("http://localhost:5001/upload-image");
        console.log("its ok");
        // "http://localhost:5001/upload-image"
        // ,formData,
        // {
        //   headers: {
        //     "Content-Type": "multipart/form-data",
        //   },
        // }
        // );

        // Handle response data here, e.g., display the processed image URL
        // console.log("Processed Image URL:", response.data.image_path);
        // alert("Image processed successfully. Check console for path!");
      } catch (error) {
        console.error("Error processing image:", error);
        alert("Failed to process image");
      }
    } else {
      alert("Please upload a file first.");
    }
  };

  return (
    <div>
      <input type="file" onChange={onFileChange} />
      <button onClick={onRestoreImage}>Restore Image</button>
    </div>
  );
}
