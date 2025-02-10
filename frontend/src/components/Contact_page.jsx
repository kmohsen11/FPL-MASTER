import "./Contact.css";
const Contact = () => {
  return (
    <div className="contact-container">
      <div className="contact-content">
        <h1>Contact Us</h1>
        <p>
          If you have any questions or would like to get in touch, please use
          the information below:
        </p>
        <ul>
          <li>
            <a href="mailto:Khaled@uni.minerva.edu">Khaled@uni.minerva.edu</a>
          </li>
          <li>
            <a
              href="https://www.linkedin.com/in/khaledahmed1911/"
              target="_blank"
              rel="noopener noreferrer"
            >
              LinkedIn
            </a>
          </li>
          <li>
            {' '}
            <a
              href="https://github.com/kmohsen11"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </li>
        </ul>
      </div>
    </div>
  )
}

export default Contact
