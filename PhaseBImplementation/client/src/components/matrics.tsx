import React from "react";

interface MetricsProps {
  psnrData: { train: number[]; val: number[] };
  ssimData: { train: number[]; val: number[] };
}

const Metrics: React.FC<MetricsProps> = ({ psnrData, ssimData }) => {
  return (
    <div className="p-4 space-y-4">
      <div>
        <h2 className="text-lg font-semibold">PSNR:</h2>
        {/* Placeholder for PSNR chart component */}
      </div>
      <div>
        <h2 className="text-lg font-semibold">SSIM:</h2>
        {/* Placeholder for SSIM chart component */}
      </div>
    </div>
  );
};

export default Metrics;
