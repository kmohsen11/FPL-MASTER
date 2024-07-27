import React from 'react';
import './contact.css';
import Navbar from './Navbar';


const Contact = () => {
  return (
    <div className="contact-container">
      
      <div className="contact-content">
        <h1>Contact Us</h1>
        <p>If you have any questions or would like to get in touch, please use the information below:</p>
        <ul>
          <li>Email: <a href="mailto:Khaled@uni.minerva.ed">Khaled@uni.minerva.ed</a></li>
          <li>LinkedIn: <a href="https://www.linkedin.com/in/khaledahmed1911/" target="_blank" rel="noopener noreferrer">LinkedIn</a></li>
          <li>GitHub: <a href="https://github.com/kmohsen11" target="_blank" rel="noopener noreferrer">GitHub</a></li>
        </ul>
      </div>
    </div>
  );
}

export default Contact;
