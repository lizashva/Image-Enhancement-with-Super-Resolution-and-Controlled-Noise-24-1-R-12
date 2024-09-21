import React from "react";
import "./joinUs.css"; // Importing CSS for styling

const JoinUs = () => {
  return (
    <div className="join-us-container">
      <div className="header-text">JOIN US</div>
      <div className="image-grid">
        <img src="../../public/image1.jpg" alt="Sample1" />
        <img src="../../public/image2.jpg" alt="Sample2" />
        <img src="../../public/image3.jpg" alt="Sample3" />
        <img src="../../public/image4.jpg" alt="Sample4" />
      </div>
      <div className="footer-text">IMAGE PROCESSING WEB APPLICATION</div>
    </div>
  );
};

export default JoinUs;
