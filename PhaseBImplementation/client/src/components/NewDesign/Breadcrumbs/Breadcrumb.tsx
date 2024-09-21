import { useNavigate } from "react-router-dom";

interface BreadcrumbProps {
  pageName: string;
}
const Breadcrumb: React.FC<BreadcrumbProps> = ({ pageName }) => {
  const navigate = useNavigate();
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between m-5">
      <h2 className="text-title-md2 font-semibold text-black dark:text-white">
        {pageName}
      </h2>

      <nav>
        <ol className="flex items-center gap-2">
          <li>
            <button
              className="font-medium"
              onClick={() => {
                navigate("/home");
              }}
            >
              Home /
            </button>
          </li>
          <li className="font-medium text-primary">{pageName}</li>
        </ol>
      </nav>
    </div>
  );
};

export default Breadcrumb;
