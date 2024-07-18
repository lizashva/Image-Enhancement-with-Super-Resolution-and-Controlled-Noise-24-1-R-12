import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav>
      <div className=" bg-gray-400 my-2 flex justify-between flex-col items-center">
        <div className="space-x-4">
          <Link
            to="/trainmodels"
            className="bg-gray-400 hover:bg-purple-400 text-white font-bold py-2 px-4 rounded inline-block text-center"
          >
            Train models
          </Link>
          <Link
            to="/restoreimage"
            className="bg-gray-400 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded inline-block text-center"
          >
            Restore an image
          </Link>
          <Link
            to="/metrics"
            className="bg-gray-400 hover:bg-green-700 text-white font-bold py-2 px-4 rounded inline-block text-center"
          >
            Metrics
          </Link>
          <Link
            to="/"
            className="bg-gray-400 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded inline-block text-center"
          >
            Home
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
