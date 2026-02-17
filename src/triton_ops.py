class TritonClient:
    def __init__(self, url, model_name, input_name, input_dtype, output_name):
        from tritonclient.grpc import InferenceServerClient, InferInput

        self._InferInput = InferInput
        self._client = InferenceServerClient(url=url, verbose=False)
        if not self._client.is_server_live():
            raise RuntimeError("Triton Server not ready")

        self.model_name = model_name
        self.input_name = input_name
        self.input_dtype = input_dtype
        self.output_name = output_name

    def infer(self, input_array):
        inputs = [self._InferInput(self.input_name, input_array.shape, self.input_dtype)]
        inputs[0].set_data_from_numpy(input_array)
        res = self._client.infer(model_name=self.model_name, inputs=inputs)
        return res.as_numpy(self.output_name)

    def close(self):
        self._client.close()
