import React from 'react';
import Breadcrumb from './NewDesign/Breadcrumbs/Breadcrumb';

const HomePage: React.FC = () => {
  return (
    <>
      <Breadcrumb pageName={`Home`} />
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex flex-col items-center py-10">
        <h1 className="text-4xl font-bold text-gray-800 dark:text-gray-100 mb-8">
          Image Restoration Showcase
        </h1>

        <div className="container mx-auto px-4">
          {/* Super Resolution Section */}
          <div className="mb-12">
            <h2 className="text-3xl font-semibold text-gray-700 dark:text-gray-200 mb-6">
              Super Resolution
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="rounded-lg shadow-lg overflow-hidden bg-white dark:bg-gray-800">
                <h3 className="text-xl font-semibold text-center py-4 bg-gray-200 dark:bg-gray-700 dark:text-gray-100">
                  Low Resolution
                </h3>
                <img
                  src="/images_home/10_128.jpg"
                  alt="Low Resolution"
                  className="w-full h-auto object-cover"
                />
              </div>
              <div className="rounded-lg shadow-lg overflow-hidden bg-white dark:bg-gray-800">
                <h3 className="text-xl font-semibold text-center py-4 bg-gray-200 dark:bg-gray-700 dark:text-gray-100">
                  High Resolution
                </h3>
                <img
                  src="/images_home/high_res_10_128.jpg"
                  alt="High Resolution"
                  className="w-full h-auto object-cover"
                />
              </div>
            </div>
          </div>

          {/* Remove Noise Section */}
          <div>
            <h2 className="text-3xl font-semibold text-gray-700 dark:text-gray-200 mb-6">
              Remove Noise
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="rounded-lg shadow-lg overflow-hidden bg-white dark:bg-gray-800">
                <h3 className="text-xl font-semibold text-center py-4 bg-gray-200 dark:bg-gray-700 dark:text-gray-100">
                  Noisy Image
                </h3>
                <img
                  src="/images_home/noisy_image.jpg"
                  alt="Noisy Image"
                  className="w-full h-auto object-cover"
                />
              </div>
              <div className="rounded-lg shadow-lg overflow-hidden bg-white dark:bg-gray-800">
                <h3 className="text-xl font-semibold text-center py-4 bg-gray-200 dark:bg-gray-700 dark:text-gray-100">
                  Noise Removed
                </h3>
                <img
                  src="/images_home/denoise_image.jpg"
                  alt="Noise Removed"
                  className="w-full h-auto object-cover"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default HomePage;
