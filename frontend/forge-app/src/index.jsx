import ForgeUI, {
  render,
  Fragment,
  Text,
  TextField,
  IssuePanel,
  useProductContext,
  useState,
  Button,
  Form,
  SectionMessage,
  Heading,
  TextArea
} from "@forge/ui";
import api from "@forge/api";
import { config } from './config';

const App = () => {
  const context = useProductContext();
  const issue = context.platformContext.issueKey;

  const [description, setDescription] = useState("");
  const [acceptanceCriteria, setAcceptanceCriteria] = useState("");
  const [testCases, setTestCases] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Determine which API endpoint to use based on environment
  const apiUrl = process.env.NODE_ENV === "production" 
    ? config.apiEndpoints.production 
    : config.apiEndpoints.development;

  const fetchIssueData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.asApp().requestJira(`/rest/api/3/issue/${issue}`);
      const data = await res.json();
      
      // Extract description
      const descriptionText = data.fields.description?.content
        ?.map(block => block.content
          ?.map(content => content.text || "")
          .join(""))
        .join("\n") || "";
      
      setDescription(descriptionText);
      
      // Try to find acceptance criteria in custom fields or description
      let acText = "";
      
      // Look for a custom field named "Acceptance Criteria"
      for (const [fieldId, fieldValue] of Object.entries(data.fields)) {
        if (fieldId.includes("customfield") && 
            data.names && 
            data.names[fieldId] && 
            data.names[fieldId].toLowerCase().includes("acceptance criteria")) {
          if (typeof fieldValue === "string") {
            acText = fieldValue;
          } else if (fieldValue?.content) {
            acText = fieldValue.content
              ?.map(block => block.content
                ?.map(content => content.text || "")
                .join(""))
              .join("\n") || "";
          }
          break;
        }
      }
      
      // If no custom field found, try to extract from description
      if (!acText) {
        const acSection = descriptionText.match(/acceptance criteria:?([\s\S]*?)(?:##|\n\n|$)/i);
        if (acSection && acSection[1]) {
          acText = acSection[1].trim();
        }
      }
      
      setAcceptanceCriteria(acText);
    } catch (err) {
      setError("Failed to fetch issue data: " + err.message);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const generateTestCases = async () => {
    setIsLoading(true);
    setError(null);
    try {
      console.log(`Using API endpoint: ${apiUrl}`);
      
      // Call the backend API using the environment-specific URL
      const response = await api.fetch(apiUrl, {
        method: "POST",
        body: JSON.stringify({ 
          description: description,
          acceptance_criteria: acceptanceCriteria 
        }),
        headers: { "Content-Type": "application/json" }
      });

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const result = await response.json();
      setTestCases(result.test_cases || result.testCases); // Handle both response formats
    } catch (err) {
      setError("Failed to generate test cases: " + err.message);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <IssuePanel>
      <Fragment>
        <Heading size="medium">AI Test Case Generator</Heading>
        <Button text="Load Issue Content" onClick={fetchIssueData} isLoading={isLoading} />
        
        {error && (
          <SectionMessage appearance="error" title="Error">
            <Text>{error}</Text>
          </SectionMessage>
        )}
        
        <Form onSubmit={generateTestCases}>
          <TextArea 
            label="Description" 
            name="description" 
            defaultValue={description} 
            onChange={setDescription} 
          />
          <TextArea 
            label="Acceptance Criteria" 
            name="acceptanceCriteria" 
            defaultValue={acceptanceCriteria} 
            onChange={setAcceptanceCriteria} 
          />
          <Button text="Generate Test Cases" isLoading={isLoading} />
        </Form>
        
        {testCases && (
          <SectionMessage title="Generated Test Cases" appearance="confirmation">
            <Text>{testCases}</Text>
          </SectionMessage>
        )}
      </Fragment>
    </IssuePanel>
  );
};

export const run = render(<App />);
