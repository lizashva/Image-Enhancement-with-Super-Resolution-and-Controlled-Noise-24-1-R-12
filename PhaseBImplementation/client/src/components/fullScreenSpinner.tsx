import React from "react";
import { ClipLoader } from "react-spinners";

const FullScreenSpinner: React.FC<{ loading: boolean }> = ({ loading }) => {
  if (!loading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="text-center">
        <ClipLoader size={150} color={"#ffffff"} loading={loading} />
        <p className="text-white mt-4">Loading...</p>
      </div>
    </div>
  );
};

export default FullScreenSpinner;
